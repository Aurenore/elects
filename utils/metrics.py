import numpy as np


def get_std_score(stats: dict, nclasses: int) -> float:
    """
    For each class, get the standard deviation of the stopping times

    Args:
    - stats: dict, containing the stopping times and the true labels
    - nclasses: int, number of classes

    Returns:
    - float, mean of the standard deviations of the stopping times for each class
    """
    true_labels = stats["targets"][:, 0]
    t_stop = stats["t_stop"].squeeze()

    stds = []
    for c in range(nclasses):
        idxs = true_labels == c
        if idxs.sum() > 1:
            std = t_stop[idxs].std()
        else:
            std = 0.0
        stds.append(std)

    return np.mean(stds)


def harmonic_mean_score(accuracy: float, earliness: float) -> float:
    return 2.0 * earliness * accuracy / (earliness + accuracy)
