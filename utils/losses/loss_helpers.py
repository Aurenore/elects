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