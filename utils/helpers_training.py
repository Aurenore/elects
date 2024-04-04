import argparse
import sys
import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Run ELECTS Early Classification training on the BavarianCrops dataset.')
    parser.add_argument('--backbonemodel', type=str, default="LSTM", choices=["LSTM", "TempCNN"], help="backbone model")
    parser.add_argument('--dataset', type=str, default="breizhcrops", choices=["bavariancrops","breizhcrops", "ghana", "southsudan","unitedstates"], help="dataset")
    parser.add_argument('--alpha', type=float, default=0.5, help="trade-off parameter of earliness and accuracy (eq 6): "
                                                                 "1=full weight on accuracy; 0=full weight on earliness")
    parser.add_argument('--epsilon', type=float, default=10, help="additive smoothing parameter that helps the "
                                                                  "model recover from too early classificaitons (eq 7)")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight_decay")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="'cuda' (GPU) or 'cpu' device to run the code. "
                                                     "defaults to 'cuda' if GPU is available, otherwise 'cpu'")
    parser.add_argument('--epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('--sequencelength', type=int, default=70, help="sequencelength of the time series. If samples are shorter, "
                                                                "they are zero-padded until this length; "
                                                                "if samples are longer, they will be undersampled")
    parser.add_argument('--hidden-dims', type=int, default=64, help="number of hidden dimensions in the backbone model")
    parser.add_argument('--batchsize', type=int, default=256, help="number of samples per batch")
    parser.add_argument('--dataroot', type=str, default=os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data"), help="directory to download the "
                                                                                 "BavarianCrops dataset (400MB)."
                                                                                 "Defaults to home directory.")
    parser.add_argument('--snapshot', type=str, default="snapshots/model.pth",
                        help="pytorch state dict snapshot file")
    parser.add_argument('--resume', action='store_true')


    args = parser.parse_args()

    if args.patience < 0:
        args.patience = None

    return args


def train_epoch(model, dataloader, optimizer, criterion, device, extra_padding_list=None):
    losses = []
    model.train()
    for batch in dataloader:
        if extra_padding_list is None:
            extra_padding_list = [0]
        for extra_padding in extra_padding_list:
            optimizer.zero_grad()
            X, y_true = batch
            X, y_true = X.to(device), y_true.to(device)
            dict_padding = {"extra_padding": extra_padding}
            log_class_probabilities, probability_stopping = model(X, **dict_padding)

            loss = criterion(log_class_probabilities, probability_stopping, y_true)

            if not loss.isnan().any():
                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().detach().numpy())

    return np.stack(losses).mean()


def test_epoch(model, dataloader, criterion, device, extra_padding_list=[0]):
    model.eval()

    stats = []
    losses = []
    slengths = []

    # sort the padding in descending order
    extra_padding_list = sorted(extra_padding_list, reverse=True)

    for ids, batch in enumerate(dataloader):
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)

        seqlengths = (X[:,:,0] != 0).sum(1)
        slengths.append(seqlengths.cpu().detach())

        # mask for sequences that are not predicted yet
        unpredicted_seq_mask = torch.ones(X.shape[0], dtype=bool).to(device) 
        
        # by default, we predict the sequence with the smallest padding
        extra_padding = extra_padding_list[-1]
        dict_padding = {"extra_padding": extra_padding}

        # predict the sequence with the smallest padding
        log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X, **dict_padding)
        loss, stat = criterion(log_class_probabilities, probability_stopping, y_true, return_stats=True)
            
        if len(extra_padding_list) > 1:
            i=0 # index for the extra_padding_list
            while unpredicted_seq_mask.any() and i < len(extra_padding_list)-1:
                extra_padding = extra_padding_list[i]
                dict_padding = {"extra_padding": extra_padding}
                log_class_probabilities_temp, probability_stopping_temp, predictions_at_t_stop_temp, t_stop_temp = model.predict(X, **dict_padding)
                loss_temp, stat_temp = criterion(log_class_probabilities, probability_stopping, y_true, return_stats=True)
                # update the mask if t_stop is different from the length of the sequence (i.e. the sequence is predicted before its end)
                unpredicted_seq_mask = unpredicted_seq_mask*(t_stop >= seqlengths-extra_padding)
            
                # update the metrics data with the mask of predicted sequences
                log_class_probabilities = torch.where(~unpredicted_seq_mask.unsqueeze(1).unsqueeze(-1), log_class_probabilities_temp, log_class_probabilities)
                probability_stopping = torch.where(~unpredicted_seq_mask.unsqueeze(1), probability_stopping_temp, probability_stopping)
                predictions_at_t_stop = torch.where(~unpredicted_seq_mask, predictions_at_t_stop_temp, predictions_at_t_stop)
                t_stop = torch.where(~unpredicted_seq_mask, t_stop_temp, t_stop)
                loss = torch.where(~unpredicted_seq_mask, loss_temp, loss)
                stat = {k: np.where(~np.expand_dims(unpredicted_seq_mask.cpu().numpy(), axis=1), v, stat[k]) for k, v in stat_temp.items()}
                i+=1
            # mean: so not exact measure, as it takes into account predictions that are not kept at the end. Only indicative. 
            stat["classification_loss"] = stat["classification_loss"].mean() 
            stat["earliness_reward"] = stat["earliness_reward"].mean()
            loss = loss.mean()
        stat["loss"] = loss.cpu().detach().numpy()
        stat["probability_stopping"] = probability_stopping.cpu().detach().numpy()
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
    stats["classification_earliness"] = np.mean(stats["t_stop"].flatten()/stats["seqlengths"])

    return np.stack(losses).mean(), stats