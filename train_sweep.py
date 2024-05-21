import sys
import os 
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
import sklearn.metrics
import pandas as pd
import wandb
from utils.plots import plot_label_distribution_datasets, boxplot_stopping_times
from utils.doy import get_doys_dict_test, get_doy_stop, create_sorted_doys_dict_test, get_approximated_doys_dict
from utils.helpers_training import parse_args_sweep, train_epoch
from utils.helpers_testing import test_epoch
from utils.metrics import harmonic_mean_score
from models.model_helpers import count_parameters
import matplotlib.pyplot as plt

def main():
    # ----------------------------- CONFIGURATION -----------------------------
    wandb.init(
        notes="ELECTS with different backbone models.",
        tags=["ELECTS", "earlyrnn", "trials", "sweep", "kp", "stopping time proximity cost"],
    )
    config = wandb.config
    # only use extra padding if tempcnn
    if config.backbonemodel == "LSTM":
        extra_padding_list = [0]
        print(f"Since LSTM is used, extra padding is set to {extra_padding_list}")
    else:
        extra_padding_list = config.extra_padding_list
    
    # if timestamps are daily (new cost function) or not
    if config.daily_timestamps: 
        alpha1, alpha2, alpha3 = sample_three_uniform_numbers()
        config.update({"sequencelength": 365,
                       "decision_head": "day",
                       "loss": "stopping_time_proximity",
                       "alpha": [alpha1, alpha2, alpha3],
                       })
    else:
        config.update({"sequencelength": 102,
                       "decision_head": "default",
                        "loss": "early_reward",
                        "alpha": 0.6,
                       })

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
        model = DailyEarlyRNN(config.backbonemodel, nclasses=nclasses, input_dim=input_dim, sequencelength=config.sequencelength, hidden_dims=config.hidden_dims).to(config.device)
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

    if config.loss == "early_reward":
        criterion = EarlyRewardLoss(alpha=config.alpha, epsilon=config.epsilon, weight=class_weights)
    elif config.loss == "stopping_time_proximity":
        criterion = StoppingTimeProximityLoss(alphas=config.alpha, weight=class_weights)
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
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=config.device, extra_padding_list=extra_padding_list)
            testloss, stats = test_epoch(model, testdataloader, criterion, config.device, extra_padding_list=extra_padding_list, return_id=test_ds.return_id, daily_timestamps=config.daily_timestamps)

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
            harmonic_mean = harmonic_mean_score(accuracy, stats["classification_earliness"])

            # ----------------------------- LOGGING -----------------------------
            train_stats.append(
                dict(
                    epoch=epoch,
                    trainloss=trainloss,
                    testloss=testloss,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    kappa=kappa,
                    elects_earliness=earliness,
                    classification_loss=classification_loss,
                    earliness_reward=earliness_reward,
                    classification_earliness=stats["classification_earliness"],
                    harmonic_mean=harmonic_mean,
                )
            )
            dict_to_wandb = {
                    "loss": {"trainloss": trainloss, "testloss": testloss},
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "fscore": fscore,
                    "kappa": kappa,
                    "elects_earliness": earliness,
                    "classification_loss": classification_loss,
                    "earliness_reward": earliness_reward,
                    "classification_earliness": stats["classification_earliness"],
                    "harmonic_mean": harmonic_mean,
                    "conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=stats["targets"][:,0], preds=stats["predictions_at_t_stop"][:,0],
                            class_names=class_names, title="Confusion Matrix")
                }
            if epoch % 5 == 1:
                fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 7))
                if config.daily_timestamps:
                    doys_stop = stats["t_stop"].squeeze()
                else: 
                    doys_dict = get_approximated_doys_dict(stats["seqlengths"], length_sorted_doy_dict_test)
                    doys_stop = get_doy_stop(stats, doys_dict)
                fig_boxplot, _ = boxplot_stopping_times(doys_stop, stats, fig_boxplot, ax_boxplot, class_names)
                dict_to_wandb["boxplot"] = wandb.Image(fig_boxplot)
                plt.close(fig_boxplot)
            
            wandb.log(dict_to_wandb)
            

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
