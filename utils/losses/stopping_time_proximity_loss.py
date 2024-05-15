import torch 
from torch import nn
from utils.losses.early_reward_loss import probability_correct_class

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
        print("cross_entropy.shape: ", cross_entropy.shape)
        classification_loss = cross_entropy.mean(0)
        print("classification_loss.shape: ", classification_loss.shape)

        # earliness reward 
        t = torch.ones(N, T, device=log_class_probabilities.device) * \
                  torch.arange(T).type(torch.FloatTensor).to(log_class_probabilities.device)
        earliness_reward = probability_correct_class(log_class_probabilities, y_true) * (1 - t / T) * (1 - timestamps_left / T)
        print("earliness_reward.shape: ", earliness_reward.shape)
        earliness_reward = earliness_reward.sum(1).mean(0)
        print("earliness_reward.shape: ", earliness_reward.shape)

        # time proximity reward 
        proximity_reward = proximity_reward(log_class_probabilities, y_true, timestamps_left)

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
        

def proximity_reward(logprobabilities, targets, timestamps_left):
    """
    """
    batchsize, sequencelength, nclasses = logprobabilities.shape

    eye = torch.eye(nclasses).to(logprobabilities.device)
    t_finals = get_t_final(timestamps_left) # shape (batchsize,)

    result = 0

    for class_idx in range(nclasses):
        # get the samples that are of the class class_idx
        class_mask = targets == class_idx # shape (batchsize, sequencelength)
        class_logprobabilities = logprobabilities[class_mask] # shape (batchsize, sequencelength)
        indices = torch.arange(class_logprobabilities.shape[0]).to(logprobabilities.device) # shape (batchsize,)
        t_finals_class = t_finals[class_mask] # shape (batchsize,)

        # get all pairs of indices of the same class
        pairs = torch.combinations(indices, with_replacement=False)
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


def get_t_final(timestamples_left):
    """
    Get the final time of the sequence, which is the index of the first 0 in the timestamps_left tensor.
    INPUT: timestamps_left: torch.Tensor of shape (N, T)
    OUTPUT: torch.Tensor of shape (N,) with the final time of the sequence
    """
    t_final = torch.argmax(timestamples_left == 0, dim=1)
    return t_final