import torch 
import torch.nn as nn
from utils.losses.early_reward_loss import probability_correct_class
from models.model_helpers import get_t_stop_from_daily_timestamps

class StoppingTimeProximityLoss(nn.Module):
    def __init__(self, alphas=[1/3, 1/3, 1/3], weight=None):
        """
        INPUT: 
        - alphas: list of 3 floats that sum to 1, the weights of the classification loss, earliness reward and proximity reward. Must be positive
        - weight: tensor of shape (C,) with the weights for each class in the classification loss
        OUTPUT: 
        - None
        """
        super(StoppingTimeProximityLoss, self).__init__()

        self.negative_log_likelihood = nn.NLLLoss(reduction="none", weight=weight)
        self.alphas = alphas

    def forward(self, log_class_probabilities, timestamps_left, y_true, return_stats=False):
        N, T, C = log_class_probabilities.shape

        # classification loss
        cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = cross_entropy.sum(1).mean(0) # sum over time, mean over batch

        # earliness reward 
        t = torch.ones(N, T, device=log_class_probabilities.device) * \
                  torch.arange(T).type(torch.FloatTensor).to(log_class_probabilities.device)
        earliness_reward = probability_correct_class(log_class_probabilities, y_true) * (1 - t / T) * (1 - timestamps_left / T)
        earliness_reward = earliness_reward.sum(1).mean(0) # sum over time, mean over batch

        # time proximity reward 
        proximity_reward = get_proximity_reward(log_class_probabilities, y_true, timestamps_left)

        # total loss
        loss = self.alphas[0] * classification_loss - self.alphas[1] * earliness_reward - self.alphas[2] * proximity_reward

        if return_stats: 
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                proximity_reward=proximity_reward.cpu().detach().numpy()
            )
            return loss, stats
        else:
            return loss
        

def get_proximity_reward(logprobabilities, targets, timestamps_left, max_number_pairs=256):
    """
    """
    batchsize, sequencelength, nclasses = logprobabilities.shape

    t_finals = get_t_stop_from_daily_timestamps(timestamps_left).to(logprobabilities.device) # shape (batchsize,)

    result = 0

    for class_idx in range(nclasses):
        # get the samples that are of the class class_idx
        class_mask = (targets[:, 0] == class_idx).to(logprobabilities.device) # shape (batchsize, )
        class_size = class_mask.sum() 
        
        indices = torch.arange(class_size).to(logprobabilities.device) # shape (class_size,)
        t_finals_class = t_finals[class_mask] # shape (class_size,)
        
        # take the log probabilities of the samples of the class class_idx
        class_logprobabilities = logprobabilities[class_mask,:,class_idx] # shape (class_size, sequencelength)

        # get all pairs of indices of the same class
        pairs = torch.combinations(indices, with_replacement=False)
        # if number of pairs higher than max_number_pairs, take a random sample of max_number_pairs pairs
        if len(pairs) > max_number_pairs:
            pairs = pairs[torch.randperm(len(pairs))[:max_number_pairs]]
            
        for pair in pairs:
            sample1_index, sample2_index = pair

            # get the final time of the samples
            t_final_1 = t_finals_class[sample1_index]
            t_final_2 = t_finals_class[sample2_index]

            # get the log probabilities of the samples at the final time
            logprob1 = class_logprobabilities[sample1_index, t_final_1]
            logprob2 = class_logprobabilities[sample2_index, t_final_2]

            # calculate the proximity reward
            result += torch.exp(logprob1) * torch.exp(logprob2) * ((t_final_1-t_final_2)/sequencelength)**2
            
    return result          


def sample_three_uniform_numbers():
    """
    Generate three random numbers that sum to 1, for parameters alphas in the loss function.
    """
    # Step 1: Generate three exponential random variables
    exp_distribution = torch.distributions.Exponential(rate=1.0)
    exp_samples = exp_distribution.sample((3,))
    
    # Step 2: Normalize the variables by their sum
    total = exp_samples.sum()
    uniform_samples = exp_samples / total
    
    return uniform_samples