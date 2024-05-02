import argparse
import sys
import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
import numpy as np

def parse_args():
    def int_list(value):
        # This function will split the string by spaces and convert each to an integer
        return [int(i) for i in value.split()]
    
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
    parser.add_argument('--extra-padding-list', type=int_list, default=[[0]], nargs='+', help="extra padding for the TempCNN model")
    parser.add_argument('--hidden-dims', type=int, default=64, help="number of hidden dimensions in the backbone model")
    parser.add_argument('--batchsize', type=int, default=256, help="number of samples per batch")
    parser.add_argument('--dataroot', type=str, default=os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data"), help="directory to download the "
                                                                                 "BavarianCrops dataset (400MB)."
                                                                                 "Defaults to home directory.")
    parser.add_argument('--snapshot', type=str, default="snapshots/model.pth",
                        help="pytorch state dict snapshot file")
    parser.add_argument('--left-padding', type=bool, default=False, help="left padding for the TempCNN model")
    parser.add_argument('--resume', action='store_true')


    args = parser.parse_args()

    if args.patience < 0:
        args.patience = None
    args.extra_padding_list = [item for sublist in args.extra_padding_list for item in sublist]
    
    return args

def parse_args_sweep():    
    parser = argparse.ArgumentParser(description='Run ELECTS Early Classification training with sweep id.')
    # parser.add_argument('--sweep-id', type=str, help="sweep id")
    parser.add_argument('--count', type=int, default=1, help="number of runs to execute per agent")
    args = parser.parse_args()
    return args


def train_epoch(model, dataloader, optimizer, criterion, device, extra_padding_list:list=[0]):
    losses = []
    model.train()
    for batch in dataloader:
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
