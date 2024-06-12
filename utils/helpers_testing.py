import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch
import numpy as np
from utils.metrics import harmonic_mean_score
import sklearn.metrics
from utils.helpers_config import set_up_config

def test_epoch(model, dataloader, criterion, device, extra_padding_list:list=[0], return_id:bool=False, daily_timestamps:bool=False, **kwargs):
    model.eval()

    stats = []
    losses = []
    slengths = []

    # sort the padding in descending order
    extra_padding_list = sorted(extra_padding_list, reverse=True)

    for ids, batch in enumerate(dataloader):
        if not return_id:
            X, y_true = batch
        else:
            X, y_true, ids = batch
            ids = ids.unsqueeze(-1)
        X, y_true = X.to(device), y_true.to(device)

        if not daily_timestamps:
            seqlengths = (X[:,:,0] != 0).sum(1)
        else:
            seqlengths = 365*torch.ones(X.shape[0], device=device)
        slengths.append(seqlengths.cpu().detach())
        
        # by default, we predict the sequence with the smallest padding
        extra_padding = extra_padding_list[-1]
        dict_padding = {"extra_padding": extra_padding, "epoch": kwargs.get("epoch", 0), "criterion_alpha": criterion.alpha}

        # predict the sequence with the smallest padding
        log_class_probabilities, stopping_criteria, predictions_at_t_stop, t_stop = model.predict(X, **dict_padding)
            
        if len(extra_padding_list) > 1:
            assert not daily_timestamps, "Extra padding is not implemented for daily_timestamps"
            # mask for sequences that are not predicted yet
            unpredicted_seq_mask = torch.ones(X.shape[0], dtype=bool).to(device)
            # index for the extra_padding_list
            i=0 
            while unpredicted_seq_mask.any() and i < len(extra_padding_list)-1:
                extra_padding = extra_padding_list[i]
                dict_padding = {"extra_padding": extra_padding}
                log_class_probabilities_temp, stopping_criteria_temp, predictions_at_t_stop_temp, t_stop_temp = model.predict(X, **dict_padding)
                if model.left_padding:
                    seqlengths_temp = seqlengths - extra_padding
                else:
                    # non-zero sequence lengths is the min between nonzero_seqlengths and sequencelength - padding 
                    seqlengths_temp = torch.min(seqlengths, torch.tensor(model.sequence_length - extra_padding, device=device))
                # update the mask if t_stop is different from the length of the padded sequence (i.e. the sequence is predicted before its end)
                unpredicted_seq_mask = unpredicted_seq_mask*(t_stop_temp > seqlengths_temp)
            
                # update the metrics data with the mask of predicted sequences
                log_class_probabilities = torch.where(~unpredicted_seq_mask.unsqueeze(1).unsqueeze(-1), log_class_probabilities_temp, log_class_probabilities)
                stopping_criteria = torch.where(~unpredicted_seq_mask.unsqueeze(1), stopping_criteria_temp, stopping_criteria)
                predictions_at_t_stop = torch.where(~unpredicted_seq_mask, predictions_at_t_stop_temp, predictions_at_t_stop)
                t_stop = torch.where(~unpredicted_seq_mask, t_stop_temp, t_stop)
                i+=1

        loss, stat = criterion(log_class_probabilities, stopping_criteria, y_true, return_stats=True, **kwargs)
        stat["loss"] = loss.cpu().detach().numpy()
        # depending of the model, we define the probability of stopping or the timestamps left as the stopping criteria
        if not daily_timestamps:
            stat["probability_stopping"] = stopping_criteria.cpu().detach().numpy()
        else: 
            stat["timestamps_left"] = stopping_criteria.cpu().detach().numpy()
        stat["class_probabilities"] = log_class_probabilities.exp().cpu().detach().numpy()
        stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["targets"] = y_true.cpu().detach().numpy()
        stat["ids"] = ids

        stats.append(stat)
        losses.append(loss.cpu().detach().numpy())

    # list of dicts to dict of lists
    stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}
    stats["seqlengths"] = torch.cat(slengths).numpy()
    
    return np.stack(losses).mean(), stats


def test_dataset(model, test_ds, criterion, device, batch_size, extra_padding_list:list=[0], return_id:bool=False, daily_timestamps:bool=False, **kwargs):
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(test_ds, batch_size=batch_size)
        loss, stats = test_epoch(model, dataloader, criterion, device, extra_padding_list=extra_padding_list, return_id=return_id, daily_timestamps=daily_timestamps, **kwargs)
    return loss, stats


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
    harmonic_mean = harmonic_mean_score(accuracy, earliness)
    test_stats = {
        "test_loss": testloss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "kappa": kappa,
        "elects_earliness": earliness,
        "classification_loss": classification_loss,
        "earliness_reward": earliness_reward,
        "harmonic_mean": harmonic_mean,
    }
    return test_stats


def get_prob_t_stop(prob_stopping):
    """ define a function that takes the prob of stopping and returns the output with the same shape, such as 
    f(d) = [d_1, d_2*(1-d_1), d_3*(1-d_1)*(1-d_2), ...] where d_i is the prob of stopping at i-th timestep
    """
    prob_t_stop = np.zeros_like(prob_stopping)
    for i in range(prob_stopping.shape[1]):
        prob_t_stop[:, i] = np.prod(1-prob_stopping[:, :i], axis=1)*prob_stopping[:, i]
    return prob_t_stop


def get_test_stats_from_model(model, test_ds, criterion, config):
    args, _ = set_up_config(config, print_comments=True)
    testloss, stats = test_dataset(model, test_ds, criterion, args.device, args.batchsize, extra_padding_list=args.extra_padding_list, \
        return_id=test_ds.return_id, daily_timestamps=args.daily_timestamps, kwargs={"epoch": args.epochs, "criterion_alpha": args.alpha_decay[1]})
    test_stats = get_test_stats(stats, testloss, args)
    return test_stats, stats