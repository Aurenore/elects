import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch
import numpy as np
from utils.metrics import harmonic_mean_score
import sklearn.metrics


def test_dataset(model, test_ds, criterion, device, batch_size):
    """
    loads model from snapshot and tests it on the test_ds dataset
    returns a dictionary of variables (probability_stopping, predictions_at_t_stop etc)
    """
    model.eval()

    with torch.no_grad():
        dataloader = DataLoader(test_ds, batch_size=batch_size)
        stats = []
        losses = []
        slengths = []
        for batch in tqdm(dataloader, leave=False):
            X, y_true, ids = batch
            X, y_true = X.to(device), y_true.to(device)

            seqlengths = (X[:,:,0] != 0).sum(1)
            slengths.append(seqlengths.cpu().detach())

            log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X)
            loss, stat = criterion(log_class_probabilities, probability_stopping, y_true, return_stats=True)
            
            # since data is padded with 0, it is possible that t_stop is after the end of sequence (negative earliness). 
            # we clip the t_stop to the maximum sequencelength here 
            msk = t_stop > seqlengths
            t_stop[msk] = seqlengths[msk]
                        
            stat["loss"] = loss.cpu().detach().numpy()
            stat["probability_stopping"] = probability_stopping.cpu().detach().numpy()
            stat["class_probabilities"] = log_class_probabilities.exp().cpu().detach().numpy()
            stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
            stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
            stat["targets"] = y_true.cpu().detach().numpy()
            stat["ids"] = ids.unsqueeze(1)

            stats.append(stat)
            losses.append(loss.cpu().detach().numpy())

        # list of dicts to dict of lists
        stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}
        stats["seqlengths"] = torch.cat(slengths).numpy()
        stats["classification_earliness"] = np.mean(stats["t_stop"].flatten()/stats["seqlengths"])

        return np.stack(losses).mean(), stats


def get_test_stats(stats, testloss, args):
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
    test_stats = {
        "test_loss": testloss,
        "classification_loss": classification_loss,
        "earliness_reward": earliness_reward,
        "earliness": earliness,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "kappa": kappa,
        "harmonic_mean": harmonic_mean,
    }
    return test_stats