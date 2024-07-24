import sys 
import os 
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.random_numbers import sample_three_uniform_numbers


class TestMus(): 
    class Config():
        def __init__(self):
            self.alpha = 0.9
            self.backbonemodel = "LSTM"
            self.batchsize = 256
            self.corrected = True
            self.dataroot = os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data")
            self.dataset = "breizhcrops"
            self.device = "cuda"
            self.epochs = 100
            self.epsilon = 10
            self.hidden_dims = 64
            self.learning_rate = 0.001
            self.loss_weight = "balanced"
            self.patience = 30
            self.resume = False
            self.sequencelength = 365
            self.validation_set = "valid"
            self.weight_decay = 0
            self.daily_timestamps = True
            self.original_time_serie_lengths = [102]
            self.loss = "daily_reward_piecewise_lin_regr"
            self.day_head_init_bias = 5
            self.decision_head = "day"
            self.start_decision_head_training = 0
            self.alpha_decay = [0.9, 0.6]
            self.percentage_earliness_reward = 0.9
            self.mu = 150.
            
        def update(self, dict_args: dict):
            for key, value in dict_args.items():
                setattr(self, key, value)
                
    def test_sample_three_uniform_numbers(self):
        numbers = sample_three_uniform_numbers()
        assert len(numbers) == 3
        for number in numbers:
            assert 0 <= number <= 1
        assert np.isclose(numbers.sum(), 1)        
        
        # check if value error for total_sum<0
        try:
            sample_three_uniform_numbers(-1)
        except ValueError as e:
            assert str(e) == "total_sum should be greater than 0."
        
        # check if value error for total_sum=0
        try:
            sample_three_uniform_numbers(0)
        except ValueError as e:
            assert str(e) == "total_sum should be greater than 0."
        
        # check if total_sum is working
        total_sum = 5
        numbers = sample_three_uniform_numbers(total_sum)
        assert np.isclose(numbers.sum(), total_sum)
        
        total_sum = 0.7
        numbers = sample_three_uniform_numbers(total_sum)
        assert np.isclose(numbers.sum(), total_sum)
        
        
    
        


        