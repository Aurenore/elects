import torch.distributions

def sample_three_uniform_numbers(total_sum: float=1.0) -> torch.Tensor:
    """
    Generate three random numbers that sum to 'total_sum', for parameters alphas in the loss function.
    
    Args:
        total_sum (float): the sum of the three random numbers
    
    Returns:
        uniform_samples (torch.Tensor): the three random numbers
    """
    if total_sum <= 0:
        raise ValueError("total_sum should be greater than 0.")
    # Step 1: Generate three exponential random variables
    exp_distribution = torch.distributions.Exponential(rate=1.0)
    exp_samples = exp_distribution.sample((3,))
    
    # Step 2: Normalize the variables by their sum
    total = exp_samples.sum()
    uniform_samples = exp_samples / total
    
    return uniform_samples*total_sum