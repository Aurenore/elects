import torch
from torch import nn
from utils.losses.loss_helpers import probability_correct_class

class ClassificationLoss(nn.Module):
    def __init__(self, alpha=0.5, weight=None, alpha_decay: list=None, epochs: int=100, start_decision_head_training: int=0):
        """_summary_

        Args:
            alpha (float, optional): _description_. Defaults to 0.5.
            weight (list, optional): weight for each class, shape: (nclasses). Defaults to None.
            alpha_decay (list, optional): contains [alpha_decay_max, alpha_decay_min]. Through the epochs, starts at alpha_decay_max
                        and get closer to alpha_decay_min. Defaults to None.
            epochs (int, optional): number of epochs. Defaults to 100.
            start_decision_head_training (int, optional): epoch to start training the decision head. Defaults to 0.
        """
        super(ClassificationLoss, self).__init__()

        self.weight = weight
        self.negative_log_likelihood = nn.NLLLoss(reduction="none", weight=self.weight)
        
            

    def forward(self, log_class_probabilities, timestamps_left, y_true, return_stats=False, **kwargs):
        N, T, C = log_class_probabilities.shape

        cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = cross_entropy.sum(1).mean(0)

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                timestamps_left=timestamps_left.cpu().detach().numpy(),
            )
            return classification_loss, stats
        else:
            return classification_loss
