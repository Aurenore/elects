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
            # max probabitliy in probability_stopping until the end of sequence (seqlengths)
            max_prob_until_sequence_end = [torch.max(probability_stopping[i, :seqlengths[i]]) for i in range(len(seqlengths))]
                        
            stat["loss"] = loss.cpu().detach().numpy()
            stat["probability_stopping"] = probability_stopping.cpu().detach().numpy()
            stat["class_probabilities"] = log_class_probabilities.exp().cpu().detach().numpy()
            stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
            stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
            stat["targets"] = y_true.cpu().detach().numpy()
            stat["ids"] = ids.unsqueeze(1)
            stat["max_prob_until_sequence_end"] = torch.stack(max_prob_until_sequence_end).cpu().detach().numpy()

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


def test_temp_masking(model, test_ds, criterion, device, batch_size, sequence_length, thresh_stop:float=0.95, step:int=5):
    """
    loads model from snapshot and tests it on the test_ds dataset, using temporal masking. The model stops predicting when the probability of stopping is above thresh_stop
    returns a dictionary of variables (probability_stopping, predictions_at_t_stop etc)
    """
    model.eval()

    with torch.no_grad():
        dataloader = DataLoader(test_ds, batch_size=batch_size)
        stats = []
        losses = []
        slengths = []
        for batch in tqdm(dataloader, leave=False):
            X, y_true, ids = batch # X.shape : (batchsize, sequence_length, input_dim), y_true.shape : (batchsize, sequence_length)
            X, y_true = X.to(device), y_true.to(device)

            seqlengths = (X[:,:,0] != 0).sum(1)
            slengths.append(seqlengths.cpu().detach())

            # initialize the temporal mask as the same shape as X, with all values set to 0 (no masking)
            temporal_mask = torch.zeros_like(X).to(device)
            unpredicted_seq_mask = torch.ones(X.shape[0], dtype=bool).to(device) # mask for sequences that are not predicted yet
            for i in range(0, sequence_length, step):
                # set the values of the temporal mask to 1, starting from the i-th timestep
                temporal_mask[unpredicted_seq_mask,:i+1,:] = True
                X_masked = X * temporal_mask
                log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X_masked)
                max_prob_until_i = torch.max(probability_stopping[:, :i+1], dim=1)[0]#torch.gather(probability_stopping, 1, t_stop.unsqueeze(1)).squeeze()
                unpredicted_seq_mask = unpredicted_seq_mask * (max_prob_until_i < thresh_stop)
                # if i%step==0:
                #     print(f"at i={i}, number of unpredicted sequences", unpredicted_seq_mask.sum())
                # if unpredicted_seq_mask.sum() == 0:
                #     break

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