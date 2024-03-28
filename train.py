import sys
import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data import BavarianCrops, BreizhCrops, SustainbenchCrops, ModisCDL
from torch.utils.data import DataLoader
from earlyrnn import EarlyRNN
import torch
from tqdm import tqdm
from loss import EarlyRewardLoss
import sklearn.metrics
import pandas as pd
import wandb
from utils.plots import plot_label_distribution_datasets, boxplot_stopping_times
from utils.doy import get_doys_dict_test, get_doy_stop, create_sorted_doys_dict_test, get_approximated_doys_dict
from utils.helpers_training import parse_args, train_epoch, test_epoch
from utils.metrics import harmonic_mean_score
import matplotlib.pyplot as plt

def main(args):
    # ----------------------------- LOAD DATASET -----------------------------

    if args.dataset == "bavariancrops":
        dataroot = os.path.join(args.dataroot,"bavariancrops")
        nclasses = 7
        input_dim = 13
        class_weights = None
        train_ds = BavarianCrops(root=dataroot,partition="train", sequencelength=args.sequencelength)
        test_ds = BavarianCrops(root=dataroot,partition="valid", sequencelength=args.sequencelength)
        class_names = test_ds.classes
    elif args.dataset == "unitedstates":
        args.dataroot = "/data/modiscdl/"
        args.sequencelength = 24
        dataroot = args.dataroot
        nclasses = 8
        input_dim = 1
        train_ds = ModisCDL(root=dataroot,partition="train", sequencelength=args.sequencelength)
        test_ds = ModisCDL(root=dataroot,partition="valid", sequencelength=args.sequencelength)
    elif args.dataset == "breizhcrops":
        dataroot = os.path.join(args.dataroot,"breizhcrops")
        nclasses = 9
        input_dim = 13
        doys_dict_test = get_doys_dict_test(dataroot=os.path.join(args.dataroot,args.dataset))
        length_sorted_doy_dict_test = create_sorted_doys_dict_test(doys_dict_test)
        print("get train and validation data...")
        train_ds = BreizhCrops(root=dataroot,partition="train", sequencelength=args.sequencelength)
        test_ds = BreizhCrops(root=dataroot,partition="valid", sequencelength=args.sequencelength)
        class_names = test_ds.ds.classname
        print("class names:", class_names)
    elif args.dataset in ["ghana"]:
        use_s2_only = False
        average_pixel = False
        max_n_pixels = 50
        dataroot = args.dataroot
        nclasses = 4
        input_dim = 12 if use_s2_only else 19  # 12 sentinel 2 + 3 x sentinel 1 + 4 * planet
        args.epochs = 500
        args.sequencelength = 365
        train_ds = SustainbenchCrops(root=dataroot,partition="train", sequencelength=args.sequencelength,
                                     country="ghana",
                                     use_s2_only=use_s2_only, average_pixel=average_pixel,
                                     max_n_pixels=max_n_pixels)
        val_ds = SustainbenchCrops(root=dataroot,partition="val", sequencelength=args.sequencelength,
                                    country="ghana", use_s2_only=use_s2_only, average_pixel=average_pixel,
                                    max_n_pixels=max_n_pixels)

        train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])

        test_ds = SustainbenchCrops(root=dataroot,partition="test", sequencelength=args.sequencelength,
                                    country="ghana", use_s2_only=use_s2_only, average_pixel=average_pixel,
                                    max_n_pixels=max_n_pixels)
        class_names = test_ds.classes
    elif args.dataset in ["southsudan"]:
        use_s2_only = False
        dataroot = args.dataroot
        nclasses = 4
        args.sequencelength = 365
        input_dim = 12 if use_s2_only else 19 # 12 sentinel 2 + 3 x sentinel 1 + 4 * planet
        args.epochs = 500
        train_ds = SustainbenchCrops(root=dataroot,partition="train", sequencelength=args.sequencelength, country="southsudan", use_s2_only=use_s2_only)
        val_ds = SustainbenchCrops(root=dataroot,partition="val", sequencelength=args.sequencelength, country="southsudan", use_s2_only=use_s2_only)

        train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
        test_ds = SustainbenchCrops(root=dataroot, partition="val", sequencelength=args.sequencelength,
                                   country="southsudan", use_s2_only=use_s2_only)
        class_names = test_ds.classes

    else:
        raise ValueError(f"dataset {args.dataset} not recognized")
    
    traindataloader = DataLoader(
        train_ds,
        batch_size=args.batchsize)
    testdataloader = DataLoader(
        test_ds,
        batch_size=args.batchsize)
    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    # datasets = [train_ds, test_ds]
    # sets_labels = ["Train", "Validation"]
    # fig, ax = plt.subplots(figsize=(15, 7))
    # fig, ax = plot_label_distribution_datasets(datasets, sets_labels, fig, ax, title='Label distribution', labels_names=class_names)
    # wandb.log({"label_distribution": wandb.Image(fig)})
    # plt.close(fig)
        
    # ----------------------------- SET UP MODEL -----------------------------
    model = EarlyRNN(args.backbonemodel, nclasses=nclasses, input_dim=input_dim, sequencelength=args.sequencelength, hidden_dims=args.hidden_dims).to(args.device)


    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # exclude decision head linear bias from weight decay
    decay, no_decay = list(), list()
    for name, param in model.named_parameters():
        if name == "stopping_decision_head.projection.0.bias":
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0, "lr": args.learning_rate}, {'params': decay}],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = EarlyRewardLoss(alpha=args.alpha, epsilon=args.epsilon)

    if args.resume and os.path.exists(args.snapshot):
        model.load_state_dict(torch.load(args.snapshot, map_location=args.device))
        optimizer_snapshot = os.path.join(os.path.dirname(args.snapshot),
                                          os.path.basename(args.snapshot).replace(".pth", "_optimizer.pth")
                                          )
        optimizer.load_state_dict(torch.load(optimizer_snapshot, map_location=args.device))
        df = pd.read_csv(args.snapshot + ".csv")
        train_stats = df.to_dict("records")
        start_epoch = train_stats[-1]["epoch"]
        print(f"resuming from {args.snapshot} epoch {start_epoch}")
    else:
        train_stats = []
        start_epoch = 1

    not_improved = 0
    
    # ----------------------------- TRAINING -----------------------------
    print("starting training...")
    with tqdm(range(start_epoch, args.epochs + 1)) as pbar:
        for epoch in pbar:
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=args.device)
            testloss, stats = test_epoch(model, testdataloader, criterion, args.device)

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
            earliness = 1 - (stats["t_stop"].mean() / (args.sequencelength - 1))
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
            if epoch % 10 == 1:
                fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 7))
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
                    savemsg = f"saving model to {args.snapshot}"
                    os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
                    torch.save(model.state_dict(), args.snapshot)

                    optimizer_snapshot = os.path.join(os.path.dirname(args.snapshot),
                                                        os.path.basename(args.snapshot).replace(".pth", "_optimizer.pth")
                                                        )
                    torch.save(optimizer.state_dict(), optimizer_snapshot)
                    wandb.log_artifact(args.snapshot, type="model")  

                    df.to_csv(args.snapshot + ".csv")
                    not_improved = 0 # reset early stopping counter
                else:
                    not_improved += 1 # increment early stopping counter
                    if args.patience is not None:
                        savemsg = f"early stopping in {args.patience - not_improved} epochs."
                    else:
                        savemsg = ""

            pbar.set_description(f"epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, "
                        f"accuracy {accuracy:.2f}, earliness {earliness:.2f}. "
                        f"classification loss {classification_loss:.2f}, earliness reward {earliness_reward:.2f}, harmonic mean {harmonic_mean:.2f}. {savemsg}")
            
                
            if args.patience is not None:
                if not_improved > args.patience:
                    print(f"stopping training. testloss {testloss:.2f} did not improve in {args.patience} epochs.")
                    break



if __name__ == '__main__':
    args = parse_args()
    wandb.init(
        dir="/mydata/studentanya/anya/wandb/",
        project="MasterThesis",
        notes="ELECTS with different backbone models.",
        tags=["ELECTS", args.dataset, args.backbonemodel, "earlyrnn", "trials"],
        config={
        "backbonemodel": args.backbonemodel,
        "dataset": args.dataset,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "device": args.device,
        "epochs": args.epochs,
        "sequencelength": args.sequencelength,
        "hidden_dims": args.hidden_dims,
        "batchsize": args.batchsize,
        "dataroot": args.dataroot,
        "snapshot": args.snapshot,
        "resume": args.resume,
        "architecture": "EarlyRNN",
        "optimizer": "AdamW",
        "criterion": "EarlyRewardLoss",
        }
    )
    main(args)
    wandb.finish()
