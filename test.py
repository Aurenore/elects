import sys
import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data import BavarianCrops, BreizhCrops, SustainbenchCrops, ModisCDL
from earlyrnn import EarlyRNN
import torch
from utils.losses.early_reward_loss import EarlyRewardLoss
import pandas as pd
from utils.plots import plot_label_distribution_datasets, boxplot_stopping_times
from utils.doy import get_doys_dict_test, get_doy_stop
from utils.helpers_training import parse_args
from utils.helpers_testing import test_dataset, get_test_stats
import matplotlib.pyplot as plt


def main(args):
    # ----------------------------- LOAD DATASET -----------------------------
    if args.dataset == "breizhcrops":
        dataroot = os.path.join(args.dataroot,"breizhcrops")
        nclasses = 9
        input_dim = 13
        doys_dict_test = get_doys_dict_test(dataroot=os.path.join(args.dataroot,args.dataset))
        test_ds = BreizhCrops(root=dataroot,partition="eval", sequencelength=args.sequencelength, return_id=True)
        class_names = test_ds.ds.classname
        print("class names:", class_names)
    else:
        raise ValueError(f"dataset {args.dataset} not recognized")

    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    datasets = [test_ds]
    sets_labels = ["Test"]
    fig, ax = plt.subplots(figsize=(15, 7))
    fig, ax = plot_label_distribution_datasets(datasets, sets_labels, fig, ax, title='Label distribution', labels_names=class_names)
        
    # ----------------------------- LOAD MODEL -----------------------------
    model = EarlyRNN(nclasses=nclasses, input_dim=input_dim, hidden_dims=args.hidden_dims, sequencelength=args.sequencelength).to(args.device)
    model.load_state_dict(torch.load(args.snapshot))
    print("model loaded from", args.snapshot) 
    criterion = EarlyRewardLoss(alpha=args.alpha, epsilon=args.epsilon)
    
    # ----------------------------- TEST -----------------------------
    testloss, stats = test_dataset(model, test_ds, criterion, args.device, args.batchsize)
    test_stats = get_test_stats(stats, testloss, args)
    
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 7))
    doys_stop = get_doy_stop(stats, doys_dict_test, approximated=False)
    fig_boxplot, _ = boxplot_stopping_times(doys_stop, stats, fig_boxplot, ax_boxplot, class_names)

    # ----------------------------- SAVE -----------------------------
    fig.savefig(os.path.join(os.path.dirname(args.snapshot), "label_distribution.png"), bbox_inches='tight')
    fig_boxplot.savefig(os.path.join(os.path.dirname(args.snapshot), "stopping_times.png"), bbox_inches='tight')
    test_stats_df = pd.DataFrame(test_stats, index=[0])
    test_stats_df.to_csv(os.path.join(os.path.dirname(args.snapshot), "test_stats.csv"))
    plt.show()


if __name__ == '__main__':
    # use example: 
    # python test.py --dataset breizhcrops --snapshot ./models/breizhcrops_models/elects_lstm/model.pth --sequencelength 150 --hidden-dims 64
    args = parse_args()
    main(args)
