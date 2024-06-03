import torch
from torch import nn
from utils.losses.loss_helpers import probability_correct_class

class DailyRewardLoss(nn.Module):
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
        super(DailyRewardLoss, self).__init__()

        self.weight = weight
        self.negative_log_likelihood = nn.NLLLoss(reduction="none", weight=self.weight)
        self.alpha = alpha
        
        if alpha_decay is not None:
            self.alpha_decay_max = alpha_decay[0]
            self.alpha_decay_min = alpha_decay[1]
            self.epochs = epochs
        
        self.start_decision_head_training = start_decision_head_training
            

    def forward(self, log_class_probabilities, timestamps_left, y_true, return_stats=False, **kwargs):
        N, T, C = log_class_probabilities.shape
        epoch = kwargs.get("epoch", 0)

        # equation 4, right term
        t = torch.ones(N, T, device=log_class_probabilities.device) * \
                torch.arange(T).type(torch.FloatTensor).to(log_class_probabilities.device)
        log_class_probabilities_at_t_plus_zt = log_class_prob_at_t_plus_zt(log_class_probabilities, timestamps_left)

        # compute alpha
        if hasattr(self, "alpha_decay_max"):
            if epoch >= self.start_decision_head_training:
                # alpha goes from alpha_decay_max to alpha_decay_min linearly
                self.alpha = self.alpha_decay_min + (self.alpha_decay_max - self.alpha_decay_min) * \
                    (1 - (epoch-self.start_decision_head_training)/(self.epochs-self.start_decision_head_training))
                
        # earliness reward 
        if epoch>=self.start_decision_head_training and self.alpha<1.-1e-8:
            earliness_reward = probability_correct_class(log_class_probabilities_at_t_plus_zt, y_true, weight=self.weight) * (1-t/T) * (1-timestamps_left.float()/T)
            earliness_reward = earliness_reward.sum(1).mean(0)
        else:
            # if the decision head is not trained, the earliness reward is zero
            earliness_reward = torch.tensor(0.0, device=log_class_probabilities.device) 

        # equation 4 left term
        if epoch>=self.start_decision_head_training and self.alpha<1.-1e-8:
            cross_entropy = self.negative_log_likelihood(log_class_probabilities_at_t_plus_zt.view(N*T,C), y_true.view(N*T)).view(N,T)
        else: 
            cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = cross_entropy.sum(1).mean(0)

        loss = self.alpha * classification_loss - (1-self.alpha) * earliness_reward

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                timestamps_left=timestamps_left.cpu().detach().numpy(),
            )
            return loss, stats
        else:
            return loss
        

def log_class_prob_at_t_plus_zt(log_class_probabilities, timestamps_left):
    """
    INPUT:
    - log_class_probabilities: tensor of shape (batchsize, sequencelength, nclasses)
    - timestamps_left: tensor of shape (batchsize, sequencelength)
    OUTPUT:
    - log_class_prob_at_t_plus_zt: tensor of shape (batchsize, sequence_lenght, nclasses) with the log_class_probabilities at the final timestamp, 
        i.e. t+timestamps_left, where t is the index of the log_class_probabilities.
    """
    t_plus_zt = torch.arange(log_class_probabilities.shape[1]).unsqueeze(0).to(log_class_probabilities.device) + timestamps_left.int() # shape (batchsize, sequencelength)
    # if t_plus_zt is larger than the last index, set it to the last index
    t_plus_zt = torch.min(t_plus_zt, torch.tensor(log_class_probabilities.shape[1]-1).to(t_plus_zt.device))
    result = log_class_probabilities[torch.arange(log_class_probabilities.shape[0]).unsqueeze(1), t_plus_zt, :] # shape (batchsize, nclasses)
    return result
