import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from torch.nn.modules.normalization import LayerNorm
from models.backbone_models.TempCNN import TempCNN
from models.model_helpers import get_backbone_model, get_t_stop_from_daily_timestamps
from models.heads import ClassificationHead, DecisionHeadDay



class DailyEarlyRNN(nn.Module):
    def __init__(self, backbone_model:str="LSTM", input_dim:int=13, hidden_dims:int=64, nclasses:int=7, num_rnn_layers:int=2, dropout:float=0.2, sequencelength:int=70):
        super(DailyEarlyRNN, self).__init__()
        # input transformations
        self.intransforms = nn.Sequential(
            nn.LayerNorm(input_dim), # normalization over D-dimension. T-dimension is untouched
            nn.Linear(input_dim, hidden_dims) # project to hidden_dims length
        )

        # backbone model 
        self.initialize_model(backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength)

        # Heads
        self.classification_head = ClassificationHead(hidden_dims, nclasses)
        self.stopping_decision_head = DecisionHeadDay(hidden_dims)

    def forward(self, x, **kwargs):
        if self.incremental_evaluation:
            x = self.padding(x, **kwargs)
        x = self.intransforms(x)
        output_tupple = self.backbone(x)
        if type(output_tupple) == tuple:
            outputs = output_tupple[0]
        else:
            outputs = output_tupple # shape : (batch_size, sequencelength, hidden_dims)
        log_class_probabilities = self.classification_head(outputs)
        timestamps_left_before_predictions= self.stopping_decision_head(outputs)

        return log_class_probabilities, timestamps_left_before_predictions

    @torch.no_grad()
    def predict(self, x, **kwargs):
        logprobabilities, timestamps_left = self.forward(x, **kwargs)
        batchsize, sequencelength, nclasses = logprobabilities.shape

        t_stop = get_t_stop_from_daily_timestamps(timestamps_left)

        # all predictions
        predictions = logprobabilities.argmax(-1)

        # predictions at time of stopping
        predictions_at_t_stop = predictions[torch.arange(batchsize), t_stop]

        return logprobabilities, timestamps_left, predictions_at_t_stop, t_stop
    
    def initialize_model(self, backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size=3):
        self.sequence_length = sequencelength
        self.backbone_model_name = backbone_model
        if self.backbone_model_name == "TempCNN":
            raise ValueError("TempCNN is not implemented for DailyEarlyRNN")
        else: 
            self.incremental_evaluation = False
        self.backbone = get_backbone_model(backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size)
        

if __name__ == "__main__":
    model = DailyEarlyRNN()