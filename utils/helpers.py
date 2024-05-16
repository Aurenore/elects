import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from models.earlyrnn import EarlyRNN
import torch
import numpy as np


def load_model(model_path, nclasses:int=9, input_dim:int=13, map_location:str="cpu"):
    "loads model from snapshot and returns it"
    model = EarlyRNN(nclasses=nclasses, input_dim=input_dim).to(map_location)
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()
    return model
    
    
def test(test_ds, model, map_location):
    """
    Tests loaded model on the test_ds dataset
    returns a dictionary of variables (probability_stopping, predictions_at_t_stop etc)
    """

    with torch.no_grad():
        model.eval()

        dataloader = DataLoader(test_ds, batch_size=256)
        model.eval()

        stats = []
        slengths = []
        for batch in tqdm(dataloader, leave=False):
            X, y_true, ids = batch
            X, y_true = X.to(map_location), y_true.to(map_location)

            seqlengths = (X[:,:,0] != 0).sum(1)
            log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X)
            
            # since data is padded with 0, it is possible that t_stop is after the end of sequence (negative earliness). 
            # we clip the t_stop to the maximum sequencelength here 
            msk = t_stop > seqlengths
            t_stop[msk] = seqlengths[msk]
            
            slengths.append(seqlengths.cpu().detach())
            
            stat = {}
            stat["probability_stopping"] = probability_stopping.cpu().detach().numpy()
            stat["class_probabilities"] = log_class_probabilities.exp().cpu().detach().numpy()
            stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
            stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
            stat["targets"] = y_true.cpu().detach().numpy()
            stat["ids"] = ids.unsqueeze(1)
            stats.append(stat)

        # list of dicts to dict of lists
        stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}
        stats["seqlengths"] = torch.cat(slengths).numpy()

        return stats
    