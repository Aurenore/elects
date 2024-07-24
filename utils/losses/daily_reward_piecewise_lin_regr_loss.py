import torch
from torch import nn
from utils.losses.loss_helpers import probability_correct_class, probability_wrong_class, \
    log_class_prob_at_t_plus_zt
from utils.random_numbers import sample_three_uniform_numbers

MU_DEFAULT = 150.
NB_DAYS_IN_YEAR = 365.

class DailyRewardPiecewiseLinRegrLoss(nn.Module):
    def __init__(self, alpha:float=1., weight=None, alpha_decay: list=None, epochs: int=100, start_decision_head_training: int=0, factor: str="v1", **kwargs):
        """ instantiate the loss function for the daily reward with a piecewise linear regression loss. 
            alpha_1*classification_loss - alpha_2*earliness_reward - alphas_3*wrong_pred_penalty + alpha_4*lin_regr_zt_loss
            alpha_1 + alpha_2 + alpha_3 + alpha_4 = 1
            alpha_1 decreases through the epochs, while alpha_2, alpha_3 and alpha_4 increase.
            alpha_1 decreases according to alpha_decay_max and alpha_decay_min
            alpha_2, alpha_3 and alpha_4 are set randomly during the training, such that the conditions are always satisfied.
        Args:
            alpha (float, optional): _description_. Defaults to 1.
            weight (list, optional): weight for each class, shape: (nclasses). Defaults to None.
            alpha_decay (list, optional): contains [alpha_decay_max, alpha_decay_min]. Through the epochs, alpha_1 starts at alpha_decay_max
                        and get closer to alpha_decay_min. Defaults to None.
            epochs (int, optional): number of epochs. Defaults to 100.
            start_decision_head_training (int, optional): epoch to start training the decision head (i.e. when alpha_1<1.). Defaults to 0.
            factor (str): factor for the wrong prediction penalty. Can be 
                - "v1": (z_t/T)*(1-t/T)
                - "v2": (t + z_t)/T
        """
        super(DailyRewardPiecewiseLinRegrLoss, self).__init__()
        assert factor in ["v1", "v2"], f"factor {factor} not implemented"
        self.factor = factor
        
        self.weight = weight
        self.negative_log_likelihood = nn.NLLLoss(reduction="none", weight=self.weight)
        self.alpha = alpha
        
        if alpha_decay is not None:
            self.alpha_decay_max = alpha_decay[0]
            self.alpha_decay_min = alpha_decay[1]
            self.epochs = epochs
        
        self.start_decision_head_training = start_decision_head_training
        
        # mus is a tensor of length nclasses, containing the mu for each class
        self.mu = kwargs.get("sequencelength", NB_DAYS_IN_YEAR)*MU_DEFAULT/NB_DAYS_IN_YEAR
        self.mus = kwargs.get("mus", torch.ones(len(weight))*self.mu).to(weight.device)
        self.percentage_earliness_reward = torch.tensor(kwargs.get("percentage_earliness_reward", 0.5), device=weight.device)
        
        # initialize the percentages of alpha_2, alpha_3 and alpha_4
        if ("percentages_other_alphas" in kwargs) and (kwargs["percentages_other_alphas"] is not None):
            if len(kwargs["percentages_other_alphas"]) != 3:
                raise ValueError("percentages_other_alphas should have length 3.")
            if not torch.isclose(torch.tensor(kwargs["percentages_other_alphas"]).sum(), torch.tensor(1.)):
                raise ValueError("percentages_other_alphas should sum to 1.")
            self.percentages_other_alphas = kwargs["percentages_other_alphas"]
        else: 
            self.percentages_other_alphas = sample_three_uniform_numbers()
        if not isinstance(self.percentages_other_alphas, torch.Tensor):
            self.percentages_other_alphas = torch.tensor(self.percentages_other_alphas, device=self.weight.device)
        
        # update alphas with the given alpha and the percentages of the other alphas
        self.alphas = self.update_alphas(self.alpha, weight.device)
        
    def forward(self, log_class_probabilities, timestamps_left, y_true, return_stats=False, **kwargs):
        N, T, C = log_class_probabilities.shape
        epoch = kwargs.get("epoch", 0)

        # log(yhat_{t+z_t})
        t = torch.ones(N, T, device=log_class_probabilities.device) * \
                torch.arange(T).type(torch.FloatTensor).to(log_class_probabilities.device)
        log_class_probabilities_at_t_plus_zt = log_class_prob_at_t_plus_zt(log_class_probabilities, timestamps_left)

        # compute the new alpha
        self.alphas = self.update_alphas_at_epoch(epoch, log_class_probabilities.device)
                            
        # earliness reward, wrong prediction penalty and piecewise linear regression loss
        if epoch>=self.start_decision_head_training and self.alpha<1.-1e-8:
            # equation (8)
            earliness_reward = probability_correct_class(log_class_probabilities_at_t_plus_zt, y_true, weight=self.weight) * (1-t/T) * (1-timestamps_left.float()/T)
            earliness_reward = earliness_reward.sum(1).mean(0)
            
            # equation (9) if "v1", or (10) if "v2"
            wrong_pred_penalty = self.compute_wrong_prediction_penalty(log_class_probabilities_at_t_plus_zt, timestamps_left, y_true, t, T)
            
            # equation (11)
            lin_regr_zt_loss = lin_regr_zt(t, T, self.mus, timestamps_left.float(), y_true) # (N, T)
            lin_regr_zt_loss = lin_regr_zt_loss.sum(1).mean(0) # sum over time, mean over batch
        else:
            # if the decision head is not trained, the earliness reward is zero
            earliness_reward = torch.tensor(0.0, device=log_class_probabilities.device) 
            wrong_pred_penalty = torch.tensor(0.0, device=log_class_probabilities.device)
            lin_regr_zt_loss = torch.tensor(0.0, device=log_class_probabilities.device)
            
        # classification loss, equation (7)
        cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = cross_entropy.sum(1).mean(0)

        # final loss, equation (12)
        loss = self.alphas[0]*classification_loss - self.alphas[1]*earliness_reward + self.alphas[2]*wrong_pred_penalty + self.alphas[3]*lin_regr_zt_loss

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
        
    def compute_wrong_prediction_penalty(self, log_class_probabilities_at_t_plus_zt, timestamps_left, y_true, t, T):
        """ compute the wrong prediction penalty, depending on the factor. 
        
        Args:
            log_class_probabilities_at_t_plus_zt (torch.Tensor): log class probabilities at t+z_t, shape: (N, T, C)
            timestamps_left (torch.Tensor): timestamps left, shape: (N, T)
            y_true (torch.Tensor): true labels, shape: (N, T)
            t (torch.Tensor): current time, shape: (N, T)
            T (int): total number of timestamps
        
        Returns:
            wrong_pred_penalty (torch.Tensor): wrong prediction penalty, shape: (1)
        
        """
        if self.factor == "v1":
            factor = (1.-t/T)*(1.-timestamps_left.float()/T)
        elif self.factor == "v2":
            factor = (t + timestamps_left.float())/T
        else: 
            raise ValueError(f"factor {self.factor} not implemented")
        wrong_pred_penalty = probability_wrong_class(log_class_probabilities_at_t_plus_zt, y_true, weight=self.weight) * factor
        wrong_pred_penalty = wrong_pred_penalty.sum(1).mean(0) # sum over time, mean over batch 
        return wrong_pred_penalty
        
    def update_alphas_at_epoch(self, epoch: int, device: torch.device) -> None:
        """ update the alphas at the given epoch.
            The alphas are updated according to the alpha_decay_max and alpha_decay_min.
            They decay linearly from alpha_decay_max to alpha_decay_min.
            
        Args:
            epoch (int): current epoch
            device (torch.device): device
        
        Returns:
            alphas (torch.Tensor): alphas, shape: (4)
        
        """
        alphas = self.alphas
        if hasattr(self, "alpha_decay_max"):
            if epoch >= self.start_decision_head_training:
                # alpha goes from alpha_decay_max to alpha_decay_min linearly
                self.alpha = self.alpha_decay_min + (self.alpha_decay_max - self.alpha_decay_min) * \
                    (1 - (epoch-self.start_decision_head_training)/(self.epochs-self.start_decision_head_training))
                alphas = self.update_alphas(self.alpha, device)
                self.alphas = alphas            
        return alphas
                
    def update_alphas(self, alpha: float, device: torch.device) -> None:
        """ update the alphas. 
            The first alpha is self.alpha, and the other ones are given by the percentages_other_alphas of the remaining part.
            
        Args:
            alpha (float): the first alpha
            device (torch.device): device
        
        Returns:
            alphas (torch.Tensor): alphas, shape: (4)
            
        """
        alphas = torch.tensor([alpha, *self.percentages_other_alphas*(1.-alpha)], device=self.weight.device)
        assert torch.isclose(alphas.sum(), torch.tensor(1., device=device)), "Alphas should sum to 1."
        return alphas 
    
    def update_mus(self, mus):
        self.mus = mus.clone().detach().to(device=self.mus.device)
        

def lin_regr_zt(t, T, mus, z_t, y_true):
    """ computes the piecewise linear regression loss for z_t
    
    INPUT:
    t: tensor of shape (N, T)
    T: float 
    mus: tensor of shape (C)
    z_t: tensor of shape (N, T)
    y_true: tensor of shape (N, T)
    
    OUTPUT: 
    loss: tensor of shape (N, T), 
    """
    # lin_term is either mus[y_true]-t-z_t for t<=mus[y_true], or z_t otherwise
    lin_term = torch.where(t<=mus[y_true], mus[y_true]-t-z_t, z_t)
    return (lin_term/T)**2