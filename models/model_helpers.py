import torch
import torch.nn as nn
from models.backbone_models.TempCNN import TempCNN


def count_parameters(model):
    """ Count the number of parameters in a model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_backbone_model(backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size):
    model_map = {
        "LSTM": {
            "class": nn.LSTM,
            "config": {
                "input_size": hidden_dims,
                "hidden_size": hidden_dims,
                "num_layers": num_rnn_layers,
                "bias": False,
                "batch_first": True,
                "dropout": dropout,
                "bidirectional": False
            }
        },
        "TempCNN": {
            "class": TempCNN,
            "config": {
                "input_dim": hidden_dims,
                "sequencelength": sequencelength,
                "kernel_size": kernel_size,
                "hidden_dims": hidden_dims, 
                "dropout": dropout
            }
        }
    }
    
    if backbone_model in model_map:
        model_info = model_map[backbone_model]
        return model_info["class"](**model_info["config"])
    else:
        raise ValueError(f"Backbone model {backbone_model} is not implemented yet.")
    

def get_t_stop_from_daily_timestamps(timestamps_left, threshold=5):
    # t_stop is the time at which the model stops. It is the first time the timestamps_left is strictly smaller than threshold
    batchsize, sequencelength = timestamps_left.shape
    time_smaller_than_1 = (timestamps_left < threshold).int()
    t_stop = torch.argmax(time_smaller_than_1, dim=1) # shape: (batchsize,)
    # for t_stop==0 and time_smaller_than_1==0, set t_stop to sequencelength-1
    t_stop = torch.where(t_stop == 0, torch.tensor(sequencelength-1).to(t_stop.device), t_stop)   

    return t_stop
