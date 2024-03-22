from data import LABELS_NAMES
from torch.utils.data import Dataset 
import numpy as np 


def extract_labels(dataset: Dataset):
    """
    Extract the labels from a dataset and return them as a NumPy array.
    """
    labels = []
    for _, y, _ in dataset:
        assert (y == y[0]).all()
        labels.append(y[0])
    labels = np.array(labels)
    return labels


def plot_label_distribution_datasets(datasets: list, sets_labels: list, fig, ax, title: str='Label distribution', labels_names: list=LABELS_NAMES):
    """"
    Plot the label distribution for multiple datasets
    """
    assert len(datasets) == len(sets_labels)
    width = 0.8/len(datasets)
    for i, ds in enumerate(datasets):
        print(f"Extracting labels from dataset {sets_labels[i]}.")
        labels = extract_labels(ds)
        counts = np.bincount(labels)
        ax.bar(np.arange(len(counts)) + i * width, counts, width=width, label=sets_labels[i])
    ax.set_xticks(np.arange(len(counts)))
    ax.set_title("Label distribution")
    ax.set_xticklabels(labels_names, rotation=45)
    ax.set_title(title)
    ax.set_xlabel('Label')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()


def ndvi(X):
    BANDS = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B9']
    red = X[BANDS.index("B4")]
    nir = X[BANDS.index("B8A")]
    return (nir-red)/(nir+red)

