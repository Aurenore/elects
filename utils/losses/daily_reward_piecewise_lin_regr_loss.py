import torch
from torch import nn
from utils.losses.loss_helpers import probability_correct_class, probability_wrong_class, \
    log_class_prob_at_t_plus_zt
from utils.losses.daily_reward_lin_regr_loss import DailyRewardLinRegrLoss, lin_regr_zt

class DailyRewardPiecewiseLinRegrLoss(DailyRewardLinRegrLoss):
    def __init__(self, alpha:float=1., weight=None, alpha_decay: list=None, epochs: int=100, start_decision_head_training: int=0, **kwargs):
        """ instantiate the loss function for the daily reward with a piecewise linear regression loss. 
            alpha_1*classification_loss - alpha_2*earliness_reward - alphas_3*wrong_pred_penalty + alpha_4*lin_regr_zt_loss
            alpha_1 + alpha_2 + alpha_3 + alpha_4 = 1
            alpha_1 decreases through the epochs, while alpha_2, alpha_3 and alpha_4 increase.
            alpha_1 decreases according to alpha_decay_max and alpha_decay_min
            alpha_2, alpha_3 and alpha_4 are set to 1-alpha_1/3
        Args:
            alpha (float, optional): _description_. Defaults to 1.
            weight (list, optional): weight for each class, shape: (nclasses). Defaults to None.
            alpha_decay (list, optional): contains [alpha_decay_max, alpha_decay_min]. Through the epochs, alpha_1 starts at alpha_decay_max
                        and get closer to alpha_decay_min. Defaults to None.
            epochs (int, optional): number of epochs. Defaults to 100.
            start_decision_head_training (int, optional): epoch to start training the decision head (i.e. when alpha_1<1.). Defaults to 0.
        """
        super(DailyRewardPiecewiseLinRegrLoss, self).__init__(alpha=alpha, weight=weight, alpha_decay=alpha_decay, epochs=epochs, start_decision_head_training=start_decision_head_training, **kwargs)

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
                            
        # earliness reward, wrong prediction penalty and piecewise linear regression loss
        if epoch>=self.start_decision_head_training and self.alpha<1.-1e-8:
            earliness_reward = probability_correct_class(log_class_probabilities_at_t_plus_zt, y_true, weight=self.weight) * (1-t/T) * (1-timestamps_left.float()/T)
            earliness_reward = earliness_reward.sum(1).mean(0)
            
            wrong_pred_penalty = probability_wrong_class(log_class_probabilities_at_t_plus_zt, y_true, weight=self.weight) * (1-t/T) * (timestamps_left.float()/T)
            wrong_pred_penalty = wrong_pred_penalty.sum(1).mean(0) # sum over time, mean over batch 
            
            lin_regr_zt_loss = lin_regr_zt(t, T, self.mus, timestamps_left.float(), y_true) # (N, T)
            lin_regr_zt_loss = lin_regr_zt_loss.sum(1).mean(0) # sum over time, mean over batch
        else:
            # if the decision head is not trained, the earliness reward is zero
            earliness_reward = torch.tensor(0.0, device=log_class_probabilities.device) 
            wrong_pred_penalty = torch.tensor(0.0, device=log_class_probabilities.device)
            lin_regr_zt_loss = torch.tensor(0.0, device=log_class_probabilities.device)
            
        # classification loss
        cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = cross_entropy.sum(1).mean(0)

        # final loss
        other_alpha = (1.-self.alpha)/3.
        self.alphas = torch.tensor([self.alpha, other_alpha, other_alpha, other_alpha], device=log_class_probabilities.device)
        loss = self.alphas[0]*classification_loss - self.alphas[1]*earliness_reward - self.alphas[2]*wrong_pred_penalty + self.alphas[2]*lin_regr_zt_loss

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                wrong_pred_penalty=wrong_pred_penalty.cpu().detach().numpy(),
                timestamps_left=timestamps_left.cpu().detach().numpy(),
                lin_regr_zt_loss=lin_regr_zt_loss.cpu().detach().numpy(),
            )
            return loss, stats
        else:
            return loss
