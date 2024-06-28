import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch
import numpy as np
from utils.metrics import harmonic_mean_score
import sklearn.metrics
from utils.helpers_config import set_up_config
from data import BreizhCrops
import os
import json

def test_epoch(model, dataloader, criterion, config, return_id:bool=False, **kwargs):
    model.eval()

    stats = []
    losses = []
    slengths = []

    for ids, batch in enumerate(dataloader):
        if not return_id:
            X, y_true = batch
        else:
            X, y_true, ids = batch
            ids = ids.unsqueeze(-1)
        X, y_true = X.to(config.device), y_true.to(config.device)

        if not config.daily_timestamps:
            seqlengths = (X[:,:,0] != 0).sum(1)
        else:
            seqlengths = 365*torch.ones(X.shape[0], device=config.device)
        slengths.append(seqlengths.cpu().detach())
        
        dict_pred = {"epoch": kwargs.get("epoch", 0), "criterion_alpha": criterion.alpha}

        log_class_probabilities, stopping_criteria, predictions_at_t_stop, t_stop = model.predict(X, **dict_pred)
        loss, stat = criterion(log_class_probabilities, stopping_criteria, y_true, return_stats=True, **kwargs)
        stat["loss"] = loss.cpu().detach().numpy()
        # depending of the model, we define the probability of stopping or the timestamps left as the stopping criteria
        if not config.daily_timestamps and config.loss == "early_reward":
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


def test_dataset(model, test_ds, criterion, config, **kwargs):
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(test_ds, batch_size=config.batchsize)
        loss, stats = test_epoch(model, dataloader, criterion, config, return_id=test_ds.return_id, **kwargs)
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
    args = set_up_config(config, print_comments=True)
    kwargs={"epoch": args.epochs, "criterion_alpha": args.alpha_decay[1]}
    testloss, stats = test_dataset(model, test_ds, criterion, args, **kwargs)
    test_stats = get_test_stats(stats, testloss, args)
    return test_stats, stats


def load_test_dataset(args, sequencelength_test=150):
    if args.dataset == "breizhcrops":
        dataroot = os.path.join(args.dataroot,"breizhcrops")
        input_dim = 13
        if not hasattr(args, "preload_ram"):
            args.preload_ram = True
        test_ds = BreizhCrops(root=dataroot, partition="eval", sequencelength=sequencelength_test, corrected=args.corrected, \
            daily_timestamps=args.daily_timestamps, original_time_serie_lengths=args.original_time_serie_lengths, \
            return_id=True, preload_ram=args.preload_ram)
        nclasses = test_ds.nclasses
        class_names = test_ds.labels_names
        print("class names:", class_names)
    else:
        raise ValueError(f"dataset {args.dataset} not recognized")
    return test_ds, nclasses, class_names, input_dim


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def save_test_stats(model_path, test_stats):
    test_stats_path = os.path.join(model_path, "test_stats.json")
    with open(test_stats_path, "w") as f:
        json.dump(test_stats, f, cls=NumpyEncoder)
    print("test_stats saved at ", test_stats_path)
    return test_stats_path


