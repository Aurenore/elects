import sys 
import os 
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.helpers_mu import mus_should_be_updated, update_mus_during_training, extract_mu_thresh


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
            self.extra_padding_list = [0]
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
                
    def test_update_mus(self):
        config = self.Config()
        config.loss = "daily_reward_piecewise_lin_regr"
        for epoch in range(1, 100):
            assert mus_should_be_updated(config, epoch, freq_update=5) == (epoch>=config.start_decision_head_training and epoch%5==0)
        
        config.loss = "daily_reward_lin_regr"
        for epoch in range(1, 100):
            assert mus_should_be_updated(config, epoch, freq_update=5) == (epoch>=config.start_decision_head_training and epoch%5==0)
        
        config.loss = ""
        for epoch in range(1, 100):
            assert mus_should_be_updated(config, epoch, freq_update=5) == False 
            
    def test_extract_mu_tresh(self):
        n_samples, n_days, nb_classes = 10, 20, 3
        class_prob = np.random.rand(n_samples, n_days, nb_classes)
        y_true = np.random.randint(0, nb_classes, n_samples)
        p_thresh = 1.
        mu_default = 150    
        mus = extract_mu_thresh(class_prob, y_true, p_thresh, mu_default)
        assert len(mus) == nb_classes, "Mus should have the same length as the number of classes."
        assert mus == [mu_default]*nb_classes, "Mus should be equal to mu_default."
        
        p_thresh = 0. 
        mus = extract_mu_thresh(class_prob, y_true, p_thresh, mu_default)
        assert mus == [0]*nb_classes, "Mus should be equal to 0."
        
        p_thresh = 0.5
        class_prob = np.zeros((n_samples, n_days, nb_classes))
        mus = extract_mu_thresh(class_prob, y_true, p_thresh, mu_default)
        assert mus == [mu_default]*nb_classes, "Mus should be equal to mu_default."
        
        class_prob = np.ones((n_samples, n_days, nb_classes))
        mus = extract_mu_thresh(class_prob, y_true, p_thresh, mu_default)
        assert mus == [0]*nb_classes, "Mus should be equal to 0."
        
        