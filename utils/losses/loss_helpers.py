import torch

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
    - log_class_prob_at_t_plus_zt: tensor of shape (batchsize, sequence_lenght, nclasses) with the log_class_probabilities at the final timestamp, 
        i.e. t+timestamps_left, where t is the index of the log_class_probabilities.
    """
    t_plus_zt = torch.arange(log_class_probabilities.shape[1]).unsqueeze(0).to(log_class_probabilities.device) + timestamps_left.int() # shape (batchsize, sequencelength)
    # if t_plus_zt is larger than the last index, set it to the last index
    t_plus_zt = torch.min(t_plus_zt, torch.tensor(log_class_probabilities.shape[1]-1).to(t_plus_zt.device))
    result = log_class_probabilities[torch.arange(log_class_probabilities.shape[0]).unsqueeze(1), t_plus_zt, :] # shape (batchsize, nclasses)
    return result


def probability_wrong_class(logprobabilities, targets, weight=None):
    """ Compute sum_c (1-y_c) * \hat{y}_c
    
    INPUT:
    - targets: shape (batchsize, sequencelength)
    - logprobabilities: shape (batchsize, sequencelength, nclasses)
    - weight: shape (nclasses, )
    
    OUTPUT: 
    - shape (batchsize, sequencelength)
    """
    batchsize, sequencelength, nclasses = logprobabilities.shape

    # Create a one-hot encoding of targets with the same device as logprobabilities
    eye = torch.eye(nclasses, dtype=torch.bool, device=logprobabilities.device)
    targets_one_hot = eye[targets]

    # Invert the one-hot encoding to focus on the wrong classes
    wrong_classes_mask = ~targets_one_hot

    # Exponentiate the log probabilities to get probabilities if necessary
    probabilities = logprobabilities.exp()

    # Mask the probabilities to only consider wrong classes
    # Using zero multiplication for correct classes to keep the same shape
    probabilities *= wrong_classes_mask

    # Apply weight if provided
    if weight is not None:
        probabilities *= weight.unsqueeze(0).unsqueeze(0)

    # Sum the probabilities across the class dimension
    result = probabilities.sum(dim=2)

    return result
