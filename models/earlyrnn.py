import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.model_helpers import get_backbone_model
from models.heads import ClassificationHead, DecisionHead


class EarlyRNN(nn.Module):
    """ ELECTS model """
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

    def forward(self, x, **kwargs):
        x = self.intransforms(x)
        output_tupple = self.backbone(x)
        if type(output_tupple) == tuple:
            outputs = output_tupple[0]
        else:
            outputs = output_tupple # shape : (batch_size, sequencelength, hidden_dims)
        log_class_probabilities = self.classification_head(outputs)
        probabilitiy_stopping = self.stopping_decision_head(outputs)

        return log_class_probabilities, probabilitiy_stopping

    @torch.no_grad()
    def predict(self, x, **kwargs):
        logprobabilities, deltas = self.forward(x, **kwargs)
        nonzero_seqlengths = (x[:,:,0] != 0).sum(1) # get the length of the sequence

        def sample_stop_decision(delta):
            dist = torch.stack([1 - delta, delta], dim=1)
            return torch.distributions.Categorical(dist).sample().bool()

        batchsize, sequencelength, nclasses = logprobabilities.shape

        stop = list()
        for t in range(sequencelength):
            # stop if sampled true and not stopped previously
            if t < sequencelength - 1:
                stop_now = sample_stop_decision(deltas[:, t]) # stop decision is true with probability delta
                stop.append(stop_now)
            else:
                # make sure to stop last
                last_stop = torch.ones(tuple(stop_now.shape)).bool()
                if torch.cuda.is_available():
                    last_stop = last_stop.cuda()
                stop.append(last_stop)

        # stack over the time dimension (multiple stops possible)
        stopped = torch.stack(stop, dim=1).bool() # has a true if the model decides to stop at time t

        # is only true if stopped for the first time
        first_stops = (stopped.cumsum(1) == 1) & stopped

        # time of stopping
        t_stop = first_stops.long().argmax(1) # get the index of the first stop

        # all predictions
        predictions = logprobabilities.argmax(-1)

        # predictions at time of stopping
        predictions_at_t_stop = torch.masked_select(predictions, first_stops)

        return logprobabilities, deltas, predictions_at_t_stop, t_stop
    
    def initialize_model(self, backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size):
        self.sequence_length = sequencelength
        self.backbone_model_name = backbone_model
        self.backbone = get_backbone_model(backbone_model, input_dim, hidden_dims, nclasses, num_rnn_layers, dropout, sequencelength, kernel_size)


if __name__ == "__main__":
    model = EarlyRNN()