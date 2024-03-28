def count_parameters(model):
    """ Count the number of parameters in a model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)