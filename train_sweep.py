import sys
import os 
import copy
#os.environ['MPLCONFIGDIR'] = "$HOME"
from sweeps.sweep_valid_eval import sweep_configuration
#os.environ["WANDB_DIR"] = os.path.join(os.path.dirname(__file__), "..", "wandb")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data import BreizhCrops
from torch.utils.data import DataLoader
from models.earlyrnn import EarlyRNN
from models.daily_earlyrnn import DailyEarlyRNN
import torch
from tqdm import tqdm
from utils.losses.early_reward_loss import EarlyRewardLoss
from utils.losses.stopping_time_proximity_loss import StoppingTimeProximityLoss, sample_three_uniform_numbers
from utils.losses.daily_reward_loss import DailyRewardLoss
from utils.losses.daily_reward_lin_regr_loss import DailyRewardLinRegrLoss, MU_DEFAULT
import sklearn.metrics
import pandas as pd
import wandb
from utils.plots import plot_label_distribution_datasets, boxplot_stopping_times, plot_timestamps_left, \
    plot_timestamps_left_per_class
from utils.plots_test import plot_fig_class_prob_wrt_time_with_mus
from utils.doy import get_doys_dict_test, get_doy_stop, create_sorted_doys_dict_test, get_approximated_doys_dict
from utils.helpers_training import parse_args_sweep, train_epoch
from utils.helpers_testing import test_epoch
from utils.metrics import harmonic_mean_score
from models.model_helpers import count_parameters
import matplotlib.pyplot as plt
from utils.extract_mu import extract_mu_thresh

def main():
    # ----------------------------- CONFIGURATION -----------------------------
    wandb.init(
        notes="ELECTS with new cost function",
        tags=["ELECTS", "earlyrnn", "trials", "sweep", "kp", "alphas", "with bias init", "with weight in earliness reward"],
    )
    config = wandb.config
    # only use extra padding if tempcnn
    if config.backbonemodel == "LSTM":
        extra_padding_list = [0]
        print(f"Since LSTM is used, extra padding is set to {extra_padding_list}")
    else:
        extra_padding_list = config.extra_padding_list

    # check if config.validation_set is set
    if not hasattr(config, "validation_set"):
        config.validation_set = "valid"
    # ----------------------------- LOAD DATASET -----------------------------
    if config.dataset == "breizhcrops":
        dataroot = os.path.join(config.dataroot,"breizhcrops")
        input_dim = 13
        doys_dict_test = get_doys_dict_test(dataroot=os.path.join(config.dataroot,config.dataset))
        length_sorted_doy_dict_test = create_sorted_doys_dict_test(doys_dict_test)
        print("get train and validation data...")
        train_ds = BreizhCrops(root=dataroot,partition="train", sequencelength=config.sequencelength, corrected=config.corrected, daily_timestamps=config.daily_timestamps, original_time_serie_lengths=config.original_time_serie_lengths)
        test_ds = BreizhCrops(root=dataroot,partition=config.validation_set, sequencelength=config.sequencelength, corrected=config.corrected, daily_timestamps=config.daily_timestamps, original_time_serie_lengths=config.original_time_serie_lengths)
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
    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    datasets = [train_ds, test_ds]
    sets_labels = ["Train", "Validation"]
    fig, ax = plt.subplots(figsize=(15, 7))
    fig, ax = plot_label_distribution_datasets(datasets, sets_labels, fig, ax, title='Label distribution', labels_names=class_names)
    wandb.log({"label_distribution": wandb.Image(fig)})
    plt.close(fig)
        
    # ----------------------------- SET UP MODEL -----------------------------
    if config.decision_head == "day":
        dict_model = {"start_decision_head_training": config.start_decision_head_training if hasattr(config, "start_decision_head_training") else 0,}
        model = DailyEarlyRNN(config.backbonemodel, nclasses=nclasses, input_dim=input_dim, sequencelength=config.sequencelength, hidden_dims=config.hidden_dims, day_head_init_bias=config.day_head_init_bias, **dict_model).to(config.device)
    else:
        model = EarlyRNN(config.backbonemodel, nclasses=nclasses, input_dim=input_dim, sequencelength=config.sequencelength, hidden_dims=config.hidden_dims, left_padding=config.left_padding).to(config.device)
    wandb.config.update({"nb_parameters": count_parameters(model)})

    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # exclude decision head linear bias from weight decay
    decay, no_decay = list(), list()
    for name, param in model.named_parameters():
        if name == "stopping_decision_head.projection.0.bias":
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0, "lr": config.learning_rate}, {'params': decay}],
                                    lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.loss_weight == "balanced":
        class_weights = train_ds.get_class_weights().to(config.device)
    else: 
        class_weights = None
    config.update({"class_weights": class_weights.cpu().detach().numpy() if class_weights is not None else None})

    if config.loss == "early_reward":
        criterion = EarlyRewardLoss(alpha=config.alpha, epsilon=config.epsilon, weight=class_weights)
    elif config.loss == "stopping_time_proximity":
        alpha4 = 1-config.alpha1-config.alpha2-config.alpha3
        config.update({"alpha4": alpha4})
        criterion = StoppingTimeProximityLoss(alphas=[config.alpha1, config.alpha2, config.alpha3, config.alpha4], weight=class_weights)
    elif config.loss == "daily_reward":
        criterion = DailyRewardLoss(alpha=config.alpha, weight=class_weights, alpha_decay=config.alpha_decay, epochs=config.epochs,\
            start_decision_head_training=config.start_decision_head_training if hasattr(config, "start_decision_head_training") else 0)
    elif config.loss == "daily_reward_lin_regr":
        mu = config.mu if hasattr(config, "mu") else MU_DEFAULT
        mus = torch.ones(nclasses)*mu
        dict_criterion = {"mus": mus, \
                "percentage_earliness_reward": config.percentage_earliness_reward if hasattr(config, "percentage_earliness_reward") else 0.9,}
        criterion = DailyRewardLinRegrLoss(alpha=config.alpha, weight=class_weights, alpha_decay=config.alpha_decay, epochs=config.epochs, \
            start_decision_head_training=config.start_decision_head_training if hasattr(config, "start_decision_head_training") else 0, \
            **dict_criterion)
    else: 
        print(f"loss {config.loss} not recognized, loss set to default: early_reward")
        criterion = EarlyRewardLoss(alpha=config.alpha, epsilon=config.epsilon, weight=class_weights)

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

    not_improved = 0
    
    # ----------------------------- TRAINING -----------------------------
    print("starting training...")
    with tqdm(range(start_epoch, config.epochs + 1)) as pbar:
        for epoch in pbar:
            # udpate mus 
            if config.loss == "daily_reward_lin_regr" and epoch>=config.start_decision_head_training and epoch%5==0:
                # compute the new mus from the classification probabilities
                mus = extract_mu_thresh(stats["class_probabilities"], stats["targets"][:, 0], config.p_thresh)
                criterion.update_mus(torch.tensor(mus))
                dict_results_epoch.update({"mus": mus})
                print(f"started training with earliness at epoch {epoch}. New parameter mus: \n{mus}")
            
            # train and test epoch
            dict_args = {"epoch": epoch}
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=config.device, extra_padding_list=extra_padding_list, **dict_args)
            testloss, stats = test_epoch(model, testdataloader, criterion, config.device, extra_padding_list=extra_padding_list, return_id=test_ds.return_id, daily_timestamps=config.daily_timestamps, **dict_args)

            # statistic logging and visualization...
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

            # ----------------------------- LOGGING -----------------------------
            dict_results_epoch = {
                    "epoch": epoch,
                    "trainloss": trainloss,
                    "testloss": testloss,                    
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "fscore": fscore,
                    "kappa": kappa,
                    "elects_earliness": earliness,
                    "classification_loss": classification_loss,
                    "earliness_reward": earliness_reward,
                    "harmonic_mean": harmonic_mean,
            }
            if config.loss == "stopping_time_proximity":
                dict_results_epoch.update({
                    "proximity_reward": stats["proximity_reward"].mean(),
                    "wrong_pred_penalty": stats["wrong_pred_penalty"].mean(),
                    })
            elif config.loss == "daily_reward_lin_regr":
                dict_results_epoch.update({
                    "lin_regr_zt_loss": stats["lin_regr_zt_loss"].mean(),
                    "alphas": criterion.alphas.cpu().detach().numpy(),
                    })
            train_stats.append(copy.deepcopy(dict_results_epoch))
            
            # update for wandb format
            dict_results_epoch.pop("trainloss")
            dict_results_epoch.pop("testloss")
            dict_results_epoch.update({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=stats["targets"][:,0], preds=stats["predictions_at_t_stop"][:,0],
                            class_names=class_names, title="Confusion Matrix"),
                            "loss": {"trainloss": trainloss, "testloss": testloss},
                            "alpha": criterion.alpha,},)
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
                
                # plot the timestamps left if config loss contains "daily_reward"
                if config.loss=="daily_reward":
                    fig_timestamps, ax_timestamps = plt.subplots(figsize=(15, 7))
                    fig_timestamps, _ = plot_timestamps_left(stats, ax_timestamps, fig_timestamps, epoch=epoch)
                    dict_results_epoch["timestamps_left_plot"] = wandb.Image(fig_timestamps)
                    plt.close(fig_timestamps)
                
                if config.loss == "daily_reward_lin_regr":
                    fig_timestamps, ax_timestamps = plt.subplots(figsize=(15, 7))
                    fig_timestamps, _ = plot_timestamps_left_per_class(fig_timestamps, ax_timestamps, stats, nclasses, class_names, mus, epoch=epoch)
                    dict_results_epoch["timestamps_left_plot"] = wandb.Image(fig_timestamps)
                    plt.close(fig_timestamps)
            
                    fig_prob_class, axes_prob_class = plt.subplots(figsize=(15, 7*len(class_names)), nrows=len(class_names), sharex=True)
                    fig_prob_class, _ = plot_fig_class_prob_wrt_time_with_mus(fig_prob_class, axes_prob_class, \
                            stats["class_probabilities"], stats["targets"][:, 0], class_names, mus, config.p_thresh, \
                            alpha=0.1, epoch=epoch)    
                    dict_results_epoch["class_probabilities_wrt_time"] = wandb.Image(fig_prob_class)
                    plt.close(fig_prob_class)
                    
            
            
            wandb.log(dict_results_epoch)

            df = pd.DataFrame(train_stats).set_index("epoch")

            savemsg = ""
            if len(df) > 2:
                if testloss < df.testloss[:-1].values.min():
                    savemsg = f"saving model to {config.snapshot}"
                    os.makedirs(os.path.dirname(config.snapshot), exist_ok=True)
                    torch.save(model.state_dict(), config.snapshot)

                    optimizer_snapshot = os.path.join(os.path.dirname(config.snapshot),
                                                        os.path.basename(config.snapshot).replace(".pth", "_optimizer.pth")
                                                        )
                    torch.save(optimizer.state_dict(), optimizer_snapshot)
                    wandb.log_artifact(config.snapshot, type="model")  

                    df.to_csv(config.snapshot + ".csv")
                    not_improved = 0 # reset early stopping counter
                else:
                    not_improved += 1 # increment early stopping counter
                    if config.patience is not None:
                        savemsg = f"early stopping in {config.patience - not_improved} epochs."
                    else:
                        savemsg = ""

            pbar.set_description(f"epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, "
                        f"accuracy {accuracy:.2f}, earliness {earliness:.2f}. "
                        f"classification loss {classification_loss:.2f}, earliness reward {earliness_reward:.2f}, harmonic mean {harmonic_mean:.2f}. {savemsg}")
            
                
            if config.patience is not None:
                if not_improved > config.patience:
                    print(f"stopping training. testloss {testloss:.2f} did not improve in {config.patience} epochs.")
                    break
                              
    wandb.finish()



if __name__ == '__main__':
    main()
