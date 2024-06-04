import torch
from torch import nn
from utils.losses.loss_helpers import probability_correct_class, log_class_prob_at_t_plus_zt

class DailyRewardLinRegrLoss(nn.Module):
    def __init__(self, alpha:float=1., weight=None, alpha_decay: list=None, epochs: int=100, start_decision_head_training: int=0, mu: float=150.):
        """_summary_

        Args:
            alpha (float, optional): _description_. Defaults to 0.5.
            weight (list, optional): weight for each class, shape: (nclasses). Defaults to None.
            alpha_decay (list, optional): contains [alpha_decay_max, alpha_decay_min]. Through the epochs, starts at alpha_decay_max
                        and get closer to alpha_decay_min. Defaults to None.
            epochs (int, optional): number of epochs. Defaults to 100.
            start_decision_head_training (int, optional): epoch to start training the decision head. Defaults to 0.
        """
        super(DailyRewardLinRegrLoss, self).__init__()

        self.weight = weight
        self.negative_log_likelihood = nn.NLLLoss(reduction="none", weight=self.weight)
        self.alpha = alpha
        
        if alpha_decay is not None:
            self.alpha_decay_max = alpha_decay[0]
            self.alpha_decay_min = alpha_decay[1]
            self.epochs = epochs
        
        self.start_decision_head_training = start_decision_head_training
        self.mu = mu
            

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
                            
        # earliness reward and linear regression loss
        if epoch>=self.start_decision_head_training and self.alpha<1.-1e-8:
            earliness_reward = probability_correct_class(log_class_probabilities_at_t_plus_zt, y_true, weight=self.weight) * (1-t/T) * (1-timestamps_left.float()/T)
            earliness_reward = earliness_reward.sum(1).mean(0)
            
            lin_regr_zt_loss = lin_regr_zt(t, T, self.mu, timestamps_left.float())
            lin_regr_zt_loss = lin_regr_zt_loss.sum(1).mean(0)
        else:
            # if the decision head is not trained, the earliness reward is zero
            earliness_reward = torch.tensor(0.0, device=log_class_probabilities.device) 
            lin_regr_zt_loss = torch.tensor(0.0, device=log_class_probabilities.device)
            
        # classification loss
        cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = cross_entropy.sum(1).mean(0)

        # final loss
        loss = self.alpha*classification_loss - (1.-self.alpha/2.)*earliness_reward + (1.-self.alpha/2.)*lin_regr_zt_loss

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                timestamps_left=timestamps_left.cpu().detach().numpy(),
                lin_regr_zt_loss=lin_regr_zt_loss.cpu().detach().numpy(),
            )
            return loss, stats
        else:
            return loss
        

def lin_regr_zt(t, T, mu, z_t):
    return ((mu-t-z_t)/T)**2
