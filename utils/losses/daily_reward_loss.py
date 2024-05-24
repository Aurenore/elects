import torch
from torch import nn

class DailyRewardLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=10, weight=None):
        super(DailyRewardLoss, self).__init__()

        self.negative_log_likelihood = nn.NLLLoss(reduction="none", weight=weight)
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, log_class_probabilities, timestamps_left, y_true, return_stats=False):
        N, T, C = log_class_probabilities.shape

        # equation 4, right term
        t = torch.ones(N, T, device=log_class_probabilities.device) * \
                torch.arange(T).type(torch.FloatTensor).to(log_class_probabilities.device)

        earliness_reward = Pt * probability_correct_class(log_class_probabilities, y_true) * (1 - t / T)
        earliness_reward = earliness_reward.sum(1).mean(0)

        # equation 4 left term
        cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = (cross_entropy * Pt).sum(1).mean(0)

        # equation 4
        loss = self.alpha * classification_loss - (1-self.alpha) * earliness_reward

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                probability_making_decision=Pt.cpu().detach().numpy()
            )
            return loss, stats
        else:
            return loss


def probability_correct_class(logprobabilities, targets, weight=None):
    """
    targets: shape (batchsize, sequencelength)
    logprobabilities: shape (batchsize, sequencelength, nclasses)
    weight: shape (nclasses, )
    """
    batchsize, sequencelength, nclasses = logprobabilities.shape

    eye = torch.eye(nclasses).type(torch.ByteTensor).to(logprobabilities.device)

    targets_one_hot = eye[targets]

    # implement the y*\hat{y} part of the loss function
    y_haty = torch.masked_select(logprobabilities, targets_one_hot.bool())
    result = y_haty.view(batchsize, sequencelength).exp()
    if weight is not None: 
        # for each y, in y_haty, we multiply by the weight of the class
        result = result * weight[targets]
    return result


def log_class_prob_at_t_plus_zt(log_class_probabilities, timestamps_left):
    """
    INPUT:
    - log_class_probabilities: tensor of shape (batchsize, sequencelength, nclasses)
    - timestamps_left: tensor of shape (batchsize, sequencelength)
    OUTPUT:
    - log_class_prob_at_t_plus_zt: tensor of shape (batchsize, nclasses) with the log_class_probabilities at the final timestamp, 
        i.e. t+timestamps_left, where t is the index of the log_class_probabilities.
    """
    t_plus_zt = torch.arange(log_class_probabilities.shape[1]).unsqueeze(0).to(log_class_probabilities.device) + timestamps_left # shape (batchsize, sequencelength)
    # if t_plus_zt is larger than the last index, set it to the last index
    t_plus_zt = torch.min(t_plus_zt, torch.tensor(log_class_probabilities.shape[1]-1).to(t_plus_zt.device))
    log_class_prob_at_t_plus_zt = log_class_probabilities[torch.arange(log_class_probabilities.shape[0]).unsqueeze(1), t_plus_zt, :] # shape (batchsize, nclasses)
    return log_class_prob_at_t_plus_zt
