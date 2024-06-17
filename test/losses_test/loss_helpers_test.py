import torch
import math
import sys 
import os 
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.losses.loss_helpers import probability_wrong_class

def test_probability_wrong_class():
    # Setup for test
    batchsize, sequencelength, nclasses = 2, 3, 4
    logprobabilities = torch.log(torch.rand(batchsize, sequencelength, nclasses) + 1e-6)  # Avoid log(0)
    targets = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long)
    weight = torch.ones(nclasses) / nclasses

    # Execute the function
    output = probability_wrong_class(logprobabilities, targets, weight)
    assert output.shape == (batchsize, sequencelength), "Output shape is incorrect."
    assert output.dtype == torch.float32, "Output dtype should be float32."

    # case when logprobabilities is log(1) for the first class, and the targets are the first class
    logprobabilities = torch.ones(batchsize, sequencelength, nclasses)*-1e6
    logprobabilities[:, :, 0] = math.log(1.)
    targets = torch.zeros(batchsize, sequencelength, dtype=torch.int32)
    output = probability_wrong_class(logprobabilities, targets, None)
    assert torch.allclose(output, torch.zeros(batchsize, sequencelength)), "Output should be zero, with log(1) for the first class and targets, weight=None."
    
    output = probability_wrong_class(logprobabilities, targets, weight)
    assert torch.allclose(output, torch.zeros(batchsize, sequencelength)), "Output should be zero, with log(1) for the first class and targets"
    
    # case when logprobabilities is log(1) for the first class, and the targets are the second class
    targets = torch.ones(batchsize, sequencelength, dtype=torch.int32)
    output = probability_wrong_class(logprobabilities, targets, weight=None)
    assert torch.allclose(output, torch.ones(batchsize, sequencelength)), "Output should be one."
    
    output = probability_wrong_class(logprobabilities, targets, weight)
    expected_output = torch.ones(batchsize, sequencelength)*weight[1]
    assert torch.allclose(output, expected_output), "Output should be the weights of the classes, with shape (batchsize, sequencelength)."
    