import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
#from models.EarlyClassificationModel import EarlyClassificationModel
from torch.nn.modules.normalization import LayerNorm
from models.TempCNN import TempCNN


class EarlyRNN(nn.Module):
    def __init__(self, backbone_model:str="LSTM", input_dim:int=13, hidden_dims:int=64, nclasses:int=7, num_rnn_layers:int=2, dropout:float=0.2, sequencelength:int=70, kernel_size:int=7):
        super(EarlyRNN, self).__init__()

        # input transformations
        self.intransforms = nn.Sequential(
            nn.LayerNorm(input_dim), # normalization over D-dimension. T-dimension is untouched
            nn.Linear(input_dim, hidden_dims) # project to hidden_dims length
        )

        # backbone model 
        self.initialize_model(backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size)

        # Heads
        self.classification_head = ClassificationHead(hidden_dims, nclasses)
        self.stopping_decision_head = DecisionHead(hidden_dims)

    def forward(self, x):
        x = self.intransforms(x)
        # TO DO: CORRECT THE SHAPE OF THE OUTPUTS
        # outputs, last_state_list = self.backbone(x)
        output_tupple = self.backbone(x)
        if type(output_tupple) == tuple:
            outputs = output_tupple[0]
        else:
            outputs = output_tupple
        log_class_probabilities = self.classification_head(outputs)
        probabilitiy_stopping = self.stopping_decision_head(outputs)

        return log_class_probabilities, probabilitiy_stopping

    @torch.no_grad()
    def predict(self, x):
        logprobabilities, deltas = self.forward(x)

        def sample_stop_decision(delta):
            dist = torch.stack([1 - delta, delta], dim=1)
            return torch.distributions.Categorical(dist).sample().bool()

        batchsize, sequencelength, nclasses = logprobabilities.shape

        stop = list()
        for t in range(sequencelength):
            # stop if sampled true and not stopped previously
            if t < sequencelength - 1:
                stop_now = sample_stop_decision(deltas[:, t])
                stop.append(stop_now)
            else:
                # make sure to stop last
                last_stop = torch.ones(tuple(stop_now.shape)).bool()
                if torch.cuda.is_available():
                    last_stop = last_stop.cuda()
                stop.append(last_stop)

        # stack over the time dimension (multiple stops possible)
        stopped = torch.stack(stop, dim=1).bool()

        # is only true if stopped for the first time
        first_stops = (stopped.cumsum(1) == 1) & stopped

        # time of stopping
        t_stop = first_stops.long().argmax(1)

        # all predictions
        predictions = logprobabilities.argmax(-1)

        # predictions at time of stopping
        predictions_at_t_stop = torch.masked_select(predictions, first_stops)

        return logprobabilities, deltas, predictions_at_t_stop, t_stop
    
    def initialize_model(self, backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size):
        self.backbone = get_backbone_model(backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size)
        

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
                "num_classes": nclasses,
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


class ClassificationHead(torch.nn.Module):

    def __init__(self, hidden_dims, nclasses):
        super(ClassificationHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, nclasses, bias=True),
            nn.LogSoftmax(dim=2))

    def forward(self, x):
        return self.projection(x)

class DecisionHead(torch.nn.Module):

    def __init__(self, hidden_dims):
        super(DecisionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, 1, bias=True),
            nn.Sigmoid()
        )

        # initialize bias to predict late in first epochs
        torch.nn.init.normal_(self.projection[0].bias, mean=-2e1, std=1e-1)


    def forward(self, x):
        return self.projection(x).squeeze(2)

if __name__ == "__main__":
    model = EarlyRNN()
