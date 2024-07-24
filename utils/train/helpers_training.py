
import argparse
import json
import sys
import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
import numpy as np
import pandas as pd
import wandb
import copy
from data import BreizhCrops
from torch.utils.data import DataLoader
from models.earlyrnn import EarlyRNN
from models.daily_earlyrnn import DailyEarlyRNN
import torch
from utils.losses.daily_reward_piecewise_lin_regr_loss import DailyRewardPiecewiseLinRegrLoss, MU_DEFAULT, NB_DAYS_IN_YEAR
from utils.losses.early_reward_loss import EarlyRewardLoss
import sklearn.metrics
from utils.plots import plot_label_distribution_datasets, boxplot_stopping_times, plot_timestamps_left, \
    plot_timestamps_left_per_class, create_figure_and_axes
from utils.plots_test import plot_fig_class_prob_wrt_time_with_mus
from utils.doy import get_doys_dict_test, get_doy_stop, create_sorted_doys_dict_test, get_approximated_doys_dict
from utils.metrics import harmonic_mean_score, get_std_score
from models.model_helpers import count_parameters
import matplotlib.pyplot as plt

def get_run_config():
    """ get the configuration path"""
    parser = argparse.ArgumentParser(description='Training configuration for D-ELECTS')
    parser.add_argument('--configpath', type=str, help='path to the config file, in json format', default=None, required=True)
    config_file_path = parser.parse_args().configpath
    # load config file, in json format 
    if config_file_path is not None:
        # open the json file and load the configuration
        with open(config_file_path, 'r') as f:
            run_config = json.load(f)
        print(run_config)
    else:
        ValueError("Please provide a config file")
    return run_config


def train_epoch(model, dataloader, optimizer, criterion, device, **kwargs):
    """ Train the model for one epoch
    
    Args:
        model: the model
        dataloader: the dataloader
        optimizer: the optimizer
        criterion: the loss function
        device: the device
        kwargs: additional keyword arguments, e.g. epoch
        
    Returns:
        np.stack(losses).mean(): the mean of the losses over the epoch 
    """
    losses = []
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)
        dict_model = {"epoch": kwargs.get("epoch", 0), "criterion_alpha": criterion.alpha}
        log_class_probabilities, stopping_criteria = model(X, **dict_model)
        
        loss = criterion(log_class_probabilities, stopping_criteria, y_true, **kwargs)

        if not loss.isnan().any():
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())

    return np.stack(losses).mean()

# ----------------- FUNCTIONS IN MAIN -----------------
def load_dataset(config, partition="train"):
    """ loads the dataset and returns the train and test dataloaders, as well as other dataset related information.

    Args:
        config (wandb.config): configuration object

    Raises:
        ValueError: if the dataset is not recognized, an error is raised

    Returns:
        traindataloader: dataloader for the training set
        testdataloader: dataloader for the test set
        train_ds: training dataset
        test_ds: test dataset
        nclasses: number of classes
        class_names: class names
        input_dim: input dimension
        length_sorted_doy_dict_test: sorted doys dictionary
    """
    if config.dataset == "breizhcrops":
        dataroot = os.path.join(config.dataroot,"breizhcrops")
        input_dim = 13
        doys_dict_test = get_doys_dict_test(dataroot=os.path.join(config.dataroot,config.dataset))
        length_sorted_doy_dict_test = create_sorted_doys_dict_test(doys_dict_test)
        print("get train and validation data...")
        train_ds = BreizhCrops(root=dataroot,partition=partition, sequencelength=config.sequencelength, corrected=config.corrected, \
            daily_timestamps=config.daily_timestamps, original_time_serie_lengths=config.original_time_serie_lengths)
        test_ds = BreizhCrops(root=dataroot,partition=config.validation_set, sequencelength=config.sequencelength, corrected=config.corrected, \
            daily_timestamps=config.daily_timestamps, original_time_serie_lengths=config.original_time_serie_lengths)
        nclasses = train_ds.nclasses
        class_names = train_ds.labels_names
        print("class names:", class_names)
    else:
        raise ValueError(f"dataset {config.dataset} not recognized")
    
    traindataloader = DataLoader(
        train_ds,
        batch_size=config.batchsize)
    testdataloader = DataLoader(
        test_ds,
        batch_size=config.batchsize)
    return traindataloader, testdataloader, train_ds, test_ds, nclasses, class_names, input_dim, length_sorted_doy_dict_test


def set_up_model(config, nclasses, input_dim, update_wandb: bool=True):
    """ sets up the model

    Args:
        config (wandb.config): configuration of the run
        nclasses (int): number of classes
        input_dim (int): dimension of the input
        update_wandb (bool, optional): whether to update the wandb configuration. Defaults to True.

    Returns:
        model: the model
    """
    if config.decision_head == "day":
        dict_model = {"start_decision_head_training": config.start_decision_head_training if hasattr(config, "start_decision_head_training") else 0,}
        model = DailyEarlyRNN(config.backbonemodel, nclasses=nclasses, input_dim=input_dim, sequencelength=config.sequencelength, hidden_dims=config.hidden_dims, day_head_init_bias=config.day_head_init_bias, **dict_model).to(config.device)
    else:
        model = EarlyRNN(config.backbonemodel, nclasses=nclasses, input_dim=input_dim, sequencelength=config.sequencelength, hidden_dims=config.hidden_dims).to(config.device)
    if update_wandb:
        wandb.config.update({"nb_parameters": count_parameters(model)})
    return model

def set_up_optimizer(config, model):
    """ sets up the optimizer

    Args:
        config (wandb.config): configuration of the run
        model: model to optimize

    Returns:
        optimizer (torch.optim.AdamW): the optimizer
    """
    # exclude decision head linear bias from weight decay
    decay, no_decay = list(), list()
    for name, param in model.named_parameters():
        if name == "stopping_decision_head.projection.0.bias":
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0, "lr": config.learning_rate}, {'params': decay}],
                                    lr=config.learning_rate, weight_decay=config.weight_decay)
    return optimizer

def set_up_class_weights(config, train_ds):
    """ sets up the class weights given the configuration and the training dataset
    
    Args: 
        config (wandb.config): configuration of the run, with the loss_weight attribute. 
        train_ds: training dataset
    
    Returns:
        class_weights: class weights"""
    if config.loss_weight == "balanced":
        class_weights = train_ds.get_class_weights().to(config.device)
    else: 
        class_weights = None
    config.update({"class_weights": class_weights.cpu().detach().numpy() if class_weights is not None else None}, allow_val_change=True)
    return class_weights
    
def set_up_criterion(config, class_weights, nclasses, mus: torch.tensor=None, wandb_update: bool=True):
    """ sets up the criterion
    
    Args:
        config (wandb.config): configuration of the run
        class_weights: class weights
        nclasses (int): number of classes
        mus: mus for the daily_reward_piecewise_lin_regr_loss
    
    Returns:
        criterion: the criterion
        mus: mus for the daily_reward_piecewise_lin_regr_loss
        mu: mu for the daily_reward_piecewise_lin_regr_loss
    """
    # define mus if necessary
    mu = None
    if "lin_regr" in config.loss: 
        mu = int(config.sequencelength*MU_DEFAULT/NB_DAYS_IN_YEAR)
        if mus is None:
            print(f"loss {config.loss} selected, setting mus to {mu}")
            mus = torch.ones(nclasses)*mu
        else: 
            print(f"loss {config.loss} selected, mus set to {mus}")
            mus = torch.tensor(mus, dtype=torch.float)
            
    if config.loss == "early_reward":
        criterion = EarlyRewardLoss(alpha=config.alpha, epsilon=config.epsilon, weight=class_weights)
    elif config.loss == "daily_reward_piecewise_lin_regr":
        dict_criterion = {"mus": mus,
                        "percentages_other_alphas": config.percentages_other_alphas if hasattr(config, "percentages_other_alphas") else None}
        criterion = DailyRewardPiecewiseLinRegrLoss(alpha=config.alpha, weight=class_weights, alpha_decay=config.alpha_decay, epochs=config.epochs, \
            start_decision_head_training=config.start_decision_head_training if hasattr(config, "start_decision_head_training") else 0,
            factor=config.factor, **dict_criterion)
        if wandb_update:
            wandb.config.update({"percentages_other_alphas": criterion.percentages_other_alphas.cpu().detach().numpy()})
            wandb.config.update({"percentage_alpha_1": criterion.percentages_other_alphas[0].cpu().detach().numpy()})
            wandb.config.update({"percentage_alpha_2": criterion.percentages_other_alphas[1].cpu().detach().numpy()})
            wandb.config.update({"percentage_alpha_3": criterion.percentages_other_alphas[2].cpu().detach().numpy()})
    else: 
        print(f"loss {config.loss} not recognized, loss set to default: early_reward")
        criterion = EarlyRewardLoss(alpha=config.alpha, epsilon=config.epsilon, weight=class_weights)
    return criterion, mus, mu

def set_up_resume(config, model, optimizer):
    """ sets up the resume option
    
    Args:
        config (wandb.config): configuration of the run
        model: the model
        optimizer: the optimizer
    
    Returns:
        train_stats: training statistics
        start_epoch: starting epoch
    """
    if config.resume and os.path.exists(config.snapshot):
        model.load_state_dict(torch.load(config.snapshot, map_location=config.device))
        optimizer_snapshot = os.path.join(os.path.dirname(config.snapshot),
                                            os.path.basename(config.snapshot).replace(".pth", "_optimizer.pth")
                                            )
        optimizer.load_state_dict(torch.load(optimizer_snapshot, map_location=config.device))
        df = pd.read_csv(config.snapshot + ".csv")
        train_stats = df.to_dict("records")
        start_epoch = train_stats[-1]["epoch"]
        print(f"resuming from {config.snapshot} epoch {start_epoch}")
    else:
        train_stats = []
        start_epoch = 1
    return train_stats, start_epoch


def get_metrics(stats, config, nclasses):
    """ computes the metrics from the statistics
    
    Args:
        stats: the statistics
        config: the configuration
    
    Returns:
        dict_results: the results in a dictionary. The computed metrics are: 
            - accuracy
            - precision
            - recall
            - fscore
            - kappa
            - elects_earliness (1-t/T)
            - classification_loss
            - earliness_reward
            - harmonic_mean of the accuracy and earliness
    """
    precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(
        y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0], average="macro",
        zero_division=0)
    accuracy = sklearn.metrics.accuracy_score(
        y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0])
    kappa = sklearn.metrics.cohen_kappa_score(
        stats["predictions_at_t_stop"][:, 0], stats["targets"][:, 0])

    classification_loss = stats["classification_loss"].mean()
    earliness_reward = stats["earliness_reward"].mean()
    earliness = 1 - (stats["t_stop"].mean() / (config.sequencelength - 1))
    harmonic_mean = harmonic_mean_score(accuracy, earliness)
    std_score = get_std_score(stats, nclasses)
    dict_results = {                 
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "kappa": kappa,
        "elects_earliness": earliness,
        "classification_loss": classification_loss,
        "earliness_reward": earliness_reward,
        "harmonic_mean": harmonic_mean,
        "std_score": std_score,
            }
    return dict_results

def update_dict_result_epoch(dict_results_epoch, stats, config, epoch, trainloss, testloss, criterion):
    """ updates the dict_results_epoch with the correct format for the train_stats list
    
    Args:
        dict_results_epoch: the dictionary to update
        stats: the statistics
        config: the configuration
        epoch: the epoch
        trainloss: the training loss
        testloss: the test loss
        criterion: the criterion
    
    Returns:
        dict_results_epoch: the updated dictionary
    """
    dict_results_epoch.update({
        "epoch": epoch,
        "trainloss": trainloss,
        "testloss": testloss,                    
    })
    if config.loss == "daily_reward_piecewise_lin_regr":
        dict_results_epoch.update({
            "lin_regr_zt_loss": stats["lin_regr_zt_loss"].mean(),
            "alphas": criterion.alphas.cpu().detach().numpy(),
            "wrong_pred_penalty": stats["wrong_pred_penalty"].mean(),
            })
    return dict_results_epoch

def second_update_dict_result_epoch(dict_results_epoch, stats, trainloss, testloss, criterion, class_names, mus):
    """ updates the dict_results_epoch with the correct format for wandb
    
    Args:
        dict_results_epoch: the dictionary to update
        stats: the statistics
        trainloss: the training loss
        testloss: the test loss
        criterion: the criterion
        class_names: the class names
        mus: the mus for the daily_reward_piecewise_lin_regr_loss
    
    Returns:
        dict_results_epoch: the updated dictionary
    """
    dict_results_epoch.pop("trainloss")
    dict_results_epoch.pop("testloss")
    dict_results_epoch.update({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                    y_true=stats["targets"][:,0], preds=stats["predictions_at_t_stop"][:,0],
                    class_names=class_names, title="Confusion Matrix"),
                    "loss": {"trainloss": trainloss, "testloss": testloss},
                    "alpha": criterion.alpha, "mus": mus,})
    return dict_results_epoch

def get_all_metrics(stats, config, epoch, train_stats, trainloss, testloss, criterion, class_names, mus):
    """ gets all the metrics, updates the train_stats list and the dict_results_epoch for wandb
    
    Args:
        stats: the statistics
        config: the configuration
        epoch: the epoch
        train_stats: the training statistics
        trainloss: the training loss
        testloss: the test loss
        criterion: the criterion
        class_names: the class names
        mus: the mus for the daily_reward_piecewise_lin_regr_loss
        
    Returns: 
        dict_results_epoch: the updated dictionary for wandb
        train_stats: the updated training statistics
    """
    dict_results_epoch = get_metrics(stats, config, nclasses=len(class_names))
    # first update the dict_results_epoch with the correct format for train_stats
    dict_results_epoch = update_dict_result_epoch(dict_results_epoch, stats, config, epoch, trainloss, testloss, criterion)
    train_stats.append(copy.deepcopy(dict_results_epoch))
    # update the dict_results_epoch with the correct format for wandb
    dict_results_epoch = second_update_dict_result_epoch(dict_results_epoch, stats, trainloss, testloss, criterion, class_names, mus)
    return dict_results_epoch, train_stats

def plots_during_training(epoch, stats, config, dict_results_epoch, class_names, length_sorted_doy_dict_test, mus, nclasses, sequencelength):
    """ plots the boxplot of the stopping times, every 5 epochs. 
        For the daily_reward loss, also plots the timestamps left.
        For the daily_reward_piecewise_lin_regr_loss, also plots the timestamps left (for each class) and the class probabilities wrt time.
        
    Args:
        epoch: the epoch
        stats: the statistics
        config: the configuration
        dict_results_epoch: the dictionary for wandb
        class_names: the class names
        length_sorted_doy_dict_test: the sorted doys dictionary
        mus: the mus for the daily_reward_piecewise_lin_regr_loss
        nclasses: the number of classes
        sequencelength: the sequence length
    
    Returns:
        dict_results_epoch: the updated dictionary for wandb
    """
    if epoch%5==0 or epoch==1 or epoch==config.epochs:
        fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 7))
        if config.daily_timestamps:
            doys_stop = stats["t_stop"].squeeze()
        else: 
            doys_dict = get_approximated_doys_dict(stats["seqlengths"], length_sorted_doy_dict_test)
            doys_stop = get_doy_stop(stats, doys_dict)
        fig_boxplot, _ = boxplot_stopping_times(doys_stop, stats, fig_boxplot, ax_boxplot, class_names, epoch=epoch)
        dict_results_epoch["boxplot"] = wandb.Image(fig_boxplot)
        plt.close(fig_boxplot)
        
        if "lin_regr" in config.loss:
            fig_timestamps, ax_timestamps = plt.subplots(figsize=(15, 7))
            fig_timestamps, _ = plot_timestamps_left_per_class(fig_timestamps, ax_timestamps, stats, nclasses, class_names, mus, ylim=sequencelength, epoch=epoch)
            dict_results_epoch["timestamps_left_plot"] = wandb.Image(fig_timestamps)
            plt.close(fig_timestamps)
    
            fig_prob_class, axes_prob_class = create_figure_and_axes(nclasses, n_cols=2)
            fig_prob_class, _ = plot_fig_class_prob_wrt_time_with_mus(fig_prob_class, axes_prob_class, \
                    stats["class_probabilities"], stats["targets"][:, 0], class_names, mus, config.p_thresh, \
                    alpha=0.1, epoch=epoch)    
            dict_results_epoch["class_probabilities_wrt_time"] = wandb.Image(fig_prob_class)
            plt.close(fig_prob_class)
    return dict_results_epoch

def save_model_artifact(config, model, optimizer, df):
    """ saves the model to a file
    
    Args:
        config: the configuration
        model: the model
        optimizer: the optimizer
        df: the dataframe with the training statistics
    
    Returns:
        savemsg: the message indicating that the model has been saved
    """
    savemsg = f"saving model to {config.snapshot}"
    os.makedirs(os.path.dirname(config.snapshot), exist_ok=True)
    torch.save(model.state_dict(), config.snapshot)

    optimizer_snapshot = os.path.join(os.path.dirname(config.snapshot),
                                        os.path.basename(config.snapshot).replace(".pth", "_optimizer.pth")
                                        )
    torch.save(optimizer.state_dict(), optimizer_snapshot)
    wandb.log_artifact(config.snapshot, type="model")  

    df.to_csv(config.snapshot + ".csv")
    return savemsg

def update_patience(df, testloss, config, model, optimizer, not_improved):
    """ updates the patience, saves the model if testloss improved
    
    Args:
        df: the dataframe with the training statistics
        testloss: the test loss
        config: the configuration
        model: the model
        optimizer: the optimizer
        not_improved: the counter for early stopping
    
    Returns:
        savemsg: the message indicating that the model has been saved
        not_improved: the updated counter for early stopping
    """
    savemsg = ""
    if len(df) > 2:
        if testloss < df.testloss[:-1].values.min():
            savemsg = save_model_artifact(config, model, optimizer, df)
            not_improved = 0 # reset early stopping counter
        else:
            not_improved += 1 # increment early stopping counter
            if config.patience is not None:
                savemsg = f"early stopping in {config.patience - not_improved} epochs."
            else:
                savemsg = ""
    return savemsg, not_improved

def plot_label_distribution_in_training(train_ds, test_ds, class_names):
    """ plots the label distribution in the training and test set. Done at the beginning of the training.
    
    Args:
        train_ds: the training dataset
        test_ds: the test dataset
        class_names: the class names
    
    Returns:
        None
    """
    datasets = [train_ds, test_ds]
    sets_labels = ["Train", "Validation"]
    fig, ax = plt.subplots(figsize=(15, 7))
    fig, ax = plot_label_distribution_datasets(datasets, sets_labels, fig, ax, title='Label distribution', labels_names=class_names)
    wandb.log({"label_distribution": wandb.Image(fig)})
    plt.close(fig)
    
def log_description(pbar, epoch, trainloss, testloss, dict_results_epoch, savemsg):
    """ logs the description in the progress bar
    
    Args:
        pbar: the progress bar
        epoch: the epoch
        trainloss: the training loss
        testloss: the test loss
        dict_results_epoch: the dictionary with the results
        savemsg: the message indicating that the model has been saved
        
    Returns:
        None
    """
    pbar.set_description(f'epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, '
        f'accuracy {dict_results_epoch["accuracy"]:.2f}, earliness {dict_results_epoch["elects_earliness"]:.2f}, '
        f'classification loss {dict_results_epoch["classification_loss"]:.2f}, '
        f'earliness reward {dict_results_epoch["earliness_reward"]:.2f}, '
        f'harmonic mean {dict_results_epoch["harmonic_mean"]:.2f}. {savemsg}')
