import datetime
import torch
import seaborn 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data import LABELS_NAMES
from utils.test.helpers_testing import get_prob_t_stop
import os
from utils.plots import boxplot_stopping_times, plot_timestamps_left_per_class, create_figure_and_axes
from utils.doy import get_doys_dict_test, get_doy_stop

PALETTE=sns.color_palette("colorblind")

import numpy as np 


def add_doys_lines(ax, dates_of_interest:list=["2017-05-01", "2017-06-01","2017-07-01","2017-08-01","2017-09-01"], ymax=1.1, ymin=0.):
    doys_of_interest = [datetime.datetime.strptime(d,"%Y-%m-%d").timetuple().tm_yday for d in dates_of_interest]
    for d, date in zip(doys_of_interest, dates_of_interest):
        ax.axvline(d, color="gray", ymin=ymin, ymax=ymax, linestyle="--")
        ax.text(d, ymax, date.replace("2017-0",""), ha="center", fontsize=8)


def plot(doys, class_probabilities, stop_probabilities):
    fig, axs = plt.subplots(2,1, figsize=(6,4), sharex=True)
    seaborn.despine()
    ax = axs[0]
    ax.plot(doys, class_probabilities)
    #ax.set_xlabel("day of year")
    ax.set_ylabel("class score")
    add_doys_lines(ax)

    ax = axs[1]
    ax.plot(doys, stop_probabilities)
    ax.set_xlabel("day of year")
    ax.set_ylabel("stopping probability")
    add_doys_lines(ax)
    

def plot_id(id, stats, doys_dict, model, device, test_ds):
    idx = list(stats["ids"][:,0]).index(id)

    X,y, id = test_ds[idx]
    doys = doys_dict[id]

    t_stop = stats["t_stop"][idx,0]
    doy_stop = doys[t_stop]
    date_stop = datetime.datetime(2017,1,1) + datetime.timedelta(days=int(doy_stop-1))

    msk = X[:,0]>0
    X = X[msk]
    X = X.unsqueeze(0).to(device)
    log_class_probabilities, probability_stopping = model(X)

    class_probabilities = log_class_probabilities[0].cpu().detach().exp()
    stop_probabilities = probability_stopping[0].cpu().detach()

    plot(doys, class_probabilities, stop_probabilities)
    print(id)
    y = stats["targets"][idx,0]
    print(f"class {y}")

    pred = stats["predictions_at_t_stop"][idx,0]
    print(f"pred {pred}")

    print(t_stop, doy_stop, date_stop)


def plot_doy_prob(id, stats, doys_dict, model, device, test_ds, fig, ax, color=None, alpha=1.0):
    idx = list(stats["ids"][:,0]).index(id)

    X,y, id = test_ds[idx]
    doys = doys_dict[id]

    msk = X[:,0]>0
    X = X[msk]
    X = X.unsqueeze(0).to(device)
    log_class_probabilities, probability_stopping = model(X)

    class_probabilities = log_class_probabilities[0].cpu().detach().exp()
    stop_probabilities = probability_stopping[0].cpu().detach()

    if doys.shape[0] != stop_probabilities.shape[0]:
        if doys.shape[0] > stop_probabilities.shape[0]:
            doys = doys[:stop_probabilities.shape[0]]
        else:
            stop_probabilities = stop_probabilities[:doys.shape[0]]
    
    
    ax.plot(doys, stop_probabilities, color=color, alpha=alpha)


    return fig, ax


def plot_all_doy_probs(stats, doys_dict, palette=PALETTE, nclasses=9, class_names=LABELS_NAMES, alpha=0.2, nsamples=500):
    fig, axes = plt.subplots(nclasses, figsize=(12,nclasses*3))
    seaborn.despine()
    axes[-1].set_xlabel("day of year")
    for y in range(nclasses):
        ax = axes[y]
        ax.set_title(class_names[y])
        ax.set_ylabel("stopping probability")
        #get indices of class y 
        index_list = [i for i in range(stats["targets"].shape[0]) if stats["targets"][i,0] == y]
        # choose nsamples random samples in index_list
        if nsamples < len(index_list):
            index_list = np.random.choice(index_list, nsamples, replace=False)
        # doys = [doys_dict[stats["ids"][i,0]] for i in index_list]
        # stop_probs = [stats["probability_stopping"][i] for i in index_list]
        color = palette[y]
        print("for class", y, "we have", len(index_list), "samples")
        # print("type of doys: ", type(doys))
        # print("type of stop_probs: ", type(stop_probs))
        for index in index_list: 
            doy = doys_dict[stats["ids"][index,0]]
            seq_length = stats["seqlengths"][index]
            if seq_length > doy.shape[0]:
                seq_length = doy.shape[0]
            else:
                doy = doy[:seq_length]
            stop_prob = stats["probability_stopping"][index][:seq_length]
            ax.plot(doy, stop_prob, color=color, alpha=alpha)
    fig.suptitle("Stopping probabilities for each class")
    fig.tight_layout()
    return fig, axes


def plot_confusion_matrix(y_true, y_pred, class_names, fig, ax, normalize=None):
    """
    Plots a confusion matrix with the true labels on the x-axis and predicted labels on the y-axis.
    
    Parameters:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated targets as returned by a classifier.
    class_names (list): List of class names to be plotted on the x and y axis.
    fig (matplotlib.figure.Figure): The figure object.
    ax (matplotlib.axes.Axes): The axes object.
    normalize (str): {'true', 'pred', 'all'}, default=None. Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. 
                    If None, confusion matrix will not be normalized.
    
    Returns:
    matplotlib.figure.Figure: The figure object with the confusion matrix.
    """
    
    # Compute confusion matrix
    if normalize not in {'true', 'pred', 'all', None}:
        raise ValueError("normalize must be one of {'true', 'pred', 'all'}")
    conf_mat = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    # Plot the confusion matrix
    if normalize is None: 
        fmt = 'd'
    elif normalize=="true":
        # format in percentage %, with 1 decimal
        fmt = '.1%'
    else:
        fmt = '.1f'
    sns.heatmap(conf_mat.T, annot=True, cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names, fmt=fmt)

    # Labels, title and ticks
    ax.set_xlabel('True labels')
    ax.set_ylabel('Predicted labels')
    ax.xaxis.set_ticklabels(class_names, rotation=45, ha='right')
    ax.yaxis.set_ticklabels(class_names, rotation=45)

    # Compute the accuracy of the model 
    accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))
    fig.suptitle(f"Overall accuracy {100*accuracy:.1f}%", fontsize=16)
    fig.tight_layout()
    
    return fig


def plot_probability_stopping(stats, index, ax):
    prob_t_stop = get_prob_t_stop(stats["probability_stopping"])
    ax.plot(stats["probability_stopping"][index], label="prob_stopping")
    ax.plot(prob_t_stop[index], linestyle='--', label="prob_t_stop")
    # vertical line at seqlength, with label "sequence length"
    ax.axvline(x=stats["seqlengths"][index], color='b',  label="sequence length")
    # vertical line at t_stop, with label "t_stop"
    ax.axvline(x=stats["t_stop"][index,0], color='y', label="t_stop")
    ax.legend()
    ax.set_title("Probability stopping")
    return ax


def plot_class_prob_wrt_time(fig, ax, label, class_prob, y_true, class_names, alpha=0.2, add_legend=True):
    # for this label, compute the mean probability and the std through time 
    mean_prob = class_prob[y_true == label].mean(axis=0)
    std_prob = class_prob[y_true == label].std(axis=0)
    # Loop through each dimension to plot separately
    for i in range(mean_prob.shape[1]):
        current_label = class_names[i]
        if add_legend:
            label_mean = f"mean {current_label}"
            label_std = f"std {current_label}"
        else: 
            label_mean = None
            label_std = None
        ax.plot(mean_prob[:, i], label=label_mean)
        ax.fill_between(range(mean_prob.shape[0]), 
                        mean_prob[:, i] - std_prob[:, i], 
                        mean_prob[:, i] + std_prob[:, i], 
                        alpha=alpha, label=label_std)
    ax.set_ylim(0, 1)
    ax.set_title(f"{class_names[label]} class probability")
    ax.set_xlabel("Time (day)")
    ax.set_ylabel("Probability")
    ax.grid()
    return fig, ax


def plot_fig_class_prob_wrt_time(fig, axes, class_prob, y_true, class_names, alpha=0.2):
    for label in range(len(class_names)):
        if label == len(class_names)-1:
            add_legend = True
        else: 
            add_legend = False
        plot_class_prob_wrt_time(fig, axes[label], label, class_prob, y_true, class_names, alpha, add_legend)
    fig.suptitle("Class probabilities through time", fontsize=16, y=1.)
    fig.tight_layout()
    return fig, axes


def plot_class_prob_wrt_time_one_sample(fig, ax, label, class_prob, y_true, class_names, alpha=0.2):
    class_prob_label = class_prob[y_true == label]
    nb_samples = 1
    # pick nb_samples random samples
    idx = torch.randint(0, class_prob_label.shape[0], (nb_samples,))
    # Loop through each dimension to plot separately
    for i in range(len(class_names)):
        current_label = class_names[i]
        if i==label:
            current_label = f"{current_label} (true)"
        # plot all the samples idx for class_prob_label
        ax.plot(class_prob_label[idx, :, i], alpha=alpha, label=current_label)
    ax.set_ylim(0, 1)
    ax.set_title(f"{class_names[label]} class probability")
    ax.set_xlabel("Time (day)")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid()
    return fig, ax


def plot_fig_class_prob_wrt_time_one_sample(fig, axes, class_prob, y_true, class_names, alpha=0.2):
    for label in range(len(class_names)):
        plot_class_prob_wrt_time_one_sample(fig, axes[label], label, class_prob, y_true, class_names, alpha)
    fig.suptitle("Class probabilities through time", fontsize=16, y=1.)
    fig.tight_layout()
    return fig, axes


def plot_fig_class_prob_wrt_time_with_mus(fig, axes, class_prob, y_true, class_names, mus, p_thresh, alpha=0.2, epoch=None):
    fig, axes = plot_fig_class_prob_wrt_time(fig, axes, class_prob, y_true, class_names, alpha)
    if mus is not None:
        # for each ax i, plot a vertical line at mu_i and a horizontal line at p_tresh
        for label in range(len(class_names)):
            if len(mus)>label:
                axes[label].axvline(mus[label], color="red", linestyle="--")
                axes[label].axhline(p_thresh, color="black", linestyle="--") # Add labels for mus[label] and p_thresh
                # Place text for mus[label] at the top of the axes (change 'top' to 'bottom' if you prefer it at the bottom)
                axes[label].text(mus[label], axes[label].get_ylim()[1], f'$\mu = {mus[label]}$', va='top', ha='right', color='red')
                # Place text for p_thresh at the far right of the axes (change 'right' to 'left' if you prefer it on the left side)
                axes[label].text(axes[label].get_xlim()[1], p_thresh, f'$p_{{thresh}} = {p_thresh}$', va='bottom', ha='right', color='black')
            else: 
                print("mus is not defined for label", label)
    # Add text for the epoch number at the top right corner of the figure
    if epoch is not None:
        fig.text(0.99, 0.99, f"Epoch {epoch}", transform=fig.transFigure, ha='right', va='top')
    # legend in lower right corner
    n_rows = (len(class_names) + 1) // 2  # +1 ensures that there is an extra row if an odd number of classes
    legend_ax_index = n_rows * 2 - 1
    if len(axes) > legend_ax_index:
        fig.legend(loc='center', bbox_to_anchor=axes[legend_ax_index].get_position().bounds, fancybox=True, ncol=2)
    fig.suptitle("Class probabilities through time", fontsize=16, y=1.)
    fig.tight_layout()
    return fig, axes


def plots_all_figs_at_test(args, stats, model_path, run_config, class_names, nclasses, mus, sequencelength_test):
    """ With the stats from the test, plot all the figures. 
        - stopping times: boxplot of the stopping times, 
        - the confusion matrix: with accuracy, 
        if the loss is 'daily_reward_lin_regr':
        - the timestamps left per class:  
        - the class probabilities wrt time and the timestamps left per class with mus and p_thresh.
    
    Args: 
        - args (argparse.Namespace): arguments from the config file
        - stats (dict): dictionary containing the stats from the test
        - model_path (str): path to the model
        - run_config (argparse.Namespace): arguments from the config file
        - class_names (list): list of class names
        - nclasses (int): number of classes
        - mus (list): list of mus
        - sequencelength_test (int): length of the sequence for the test dataset
    
    Returns:
        None    
    """
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 7))
    if args.daily_timestamps:
        doys_stop = stats["t_stop"].squeeze()
    else: 
        doys_dict_test = get_doys_dict_test(dataroot=os.path.join(args.dataroot,args.dataset))
        doys_stop = get_doy_stop(stats, doys_dict_test, approximated=False)
    fig_boxplot, _ = boxplot_stopping_times(doys_stop, stats, fig_boxplot, ax_boxplot, class_names)
    fig_filename = os.path.join(model_path, "boxplot_stopping_times.png")
    fig_boxplot.savefig(fig_filename)
    print("fig saved at ", fig_filename)

    fig, ax = plt.subplots(figsize=(7,7))
    fig = plot_confusion_matrix(stats["targets"][:, 0], stats["predictions_at_t_stop"].flatten(), class_names, fig, ax)
    fig_filename = os.path.join(model_path, "confusion_matrix.png")
    fig.savefig(fig_filename)
    print("fig saved at ", fig_filename)
    
    # normalized matrix 
    fig_normalized, ax_normalized = plt.subplots(figsize=(7,7))
    fig_normalized = plot_confusion_matrix(stats["targets"][:, 0], stats["predictions_at_t_stop"].flatten(), class_names, fig_normalized, ax_normalized, normalize='true')
    fig_filename = os.path.join(model_path, "confusion_matrix_normalized.png")
    fig_normalized.savefig(fig_filename)
    print("fig saved at ", fig_filename)

    if "lin_regr" in run_config.loss or run_config.loss == "daily_reward":
        fig_timestamps, ax_timestamps = plt.subplots(figsize=(15, 7))
        fig_timestamps, _ = plot_timestamps_left_per_class(fig_timestamps, ax_timestamps, stats, nclasses, class_names, mus, ylim=sequencelength_test, epoch=run_config.epochs)
        fig_filename = os.path.join(model_path, "timestamps_left_per_class.png")
        fig_timestamps.savefig(fig_filename)
        print("fig saved at ", fig_filename)
            
        fig_prob_class, axes_prob_class = create_figure_and_axes(nclasses, n_cols=2)
        # check that run_cconfig has p_thresh as attribute 
        if not hasattr(run_config, "p_thresh"):
            run_config.p_thresh = None
        fig_prob_class, _ = plot_fig_class_prob_wrt_time_with_mus(fig_prob_class, axes_prob_class, \
                stats["class_probabilities"], stats["targets"][:, 0], class_names, mus, run_config.p_thresh, \
                alpha=0.1, epoch=run_config.epochs)   
        fig_filename = os.path.join(model_path, "class_probabilities_wrt_time_with_mus.png")
        fig_prob_class.savefig(fig_filename)
        print("fig saved at ", fig_filename)
    return