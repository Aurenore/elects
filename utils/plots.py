from data import LABELS_NAMES
from torch.utils.data import Dataset 
import numpy as np 
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import warnings
PALETTE=sns.color_palette("colorblind")


def extract_labels(dataset: Dataset):
    """
    Extract the labels from a dataset and return them as a NumPy array.
    """
    labels = []
    for sample in dataset:
        y = sample[1]
        assert (y == y[0]).all()
        labels.append(y[0])
    labels = np.array(labels)
    return labels


def plot_label_distribution_datasets(datasets: list, sets_labels: list, fig, ax, title: str='Label distribution', labels_names: list=LABELS_NAMES, colors=PALETTE):
    """"
    Plot the label distribution for multiple datasets
    """
    assert len(datasets) == len(sets_labels)
    width = 0.8/len(datasets)
    for i, ds in enumerate(datasets):
        print(f"Extracting labels from dataset {sets_labels[i]}.")
        labels = extract_labels(ds)
        counts = np.bincount(labels)
        ax.bar(np.arange(len(counts)) + i * width, counts, width=width, label=sets_labels[i], color=colors[i])
    ax.set_xticks(np.arange(len(counts)))
    ax.set_title("Label distribution")
    ax.set_xticklabels(labels_names, rotation=45)
    ax.set_title(title)
    ax.set_xlabel('Label')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    return fig, ax


def boxplot_stopping_times(doy_stop, stats, fig, ax, labels_names=LABELS_NAMES, colors=PALETTE):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        doys_months = [datetime.datetime(2017,m,1).timetuple().tm_yday for m in range(1,13)]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        sns.boxplot(x=doy_stop,y=stats["targets"][:,0],orient="h",ax=ax,showfliers=False, palette=colors[:len(labels_names)])
        ax.set_yticks(range(len(labels_names)))
        ax.set_yticklabels(labels_names, fontsize=16)
        ax.set_xlabel("day of year", fontsize=16)

        ax.xaxis.grid(True)
        ax.set_xticks(doys_months)
        ax.set_xticklabels(months, ha="left")

        sns.despine(left=True)
        fig.tight_layout()

    return fig, ax


def plot_boxplot(labels, t_stops, fig, ax, label_names: list=LABELS_NAMES, tmin=None, tmax=None):
    grouped = [t_stops[labels == i] for i in np.unique(labels)]
    sns.boxplot(data=grouped, orient="h", ax=ax)
    ax.set_xlabel("t_stop")
    ax.set_ylabel("class")
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_yticklabels(label_names)
    ax.set_xlim(tmin, tmax)
    fig.tight_layout()
    return fig, ax

