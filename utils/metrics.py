import numpy as np

def get_std_score(stats, nclasses: int)->float:
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
        std = t_stop[idxs].std()
        stds.append(std)
    
    return np.mean(stds)


def harmonic_mean_score(accuracy: float, earliness:float)->float:
    return 2.*earliness*accuracy/(earliness+accuracy)