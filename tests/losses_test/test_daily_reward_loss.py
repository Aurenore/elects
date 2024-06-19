import sys
import os 
from tqdm import tqdm
import torch
import numpy as np 
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.helpers_config import set_up_config
from utils.helpers_mu import mus_should_be_updated, update_mus_during_training
from utils.helpers_training import train_epoch, set_up_model, set_up_optimizer, set_up_class_weights, \
    set_up_criterion, set_up_resume, get_all_metrics, update_patience, log_description, \
    plots_during_training, load_dataset, plot_label_distribution_in_training 
from utils.helpers_config import set_up_config
from utils.helpers_testing import test_epoch as run_test_epoch

class TestDailyRewardPiecewiseLinRegrLoss(): 
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
            self.factor = "v1"
            
        def update(self, dict_args: dict):
            for key, value in dict_args.items():
                setattr(self, key, value)
                
    def test_instantiation(self): 
        torch.autograd.set_detect_anomaly(True)

        config = self.Config()
        config, extra_padding_list = set_up_config(config)
        assert config is not None, "Config setup failed."
        assert isinstance(extra_padding_list, list), "Extra padding list should be a list."
        
        traindataloader, testdataloader, train_ds, test_ds, nclasses, class_names, input_dim, length_sorted_doy_dict_test = load_dataset(config)

        model = set_up_model(config, nclasses, input_dim, update_wandb=False)
        optimizer = set_up_optimizer(config, model)
        class_weights = set_up_class_weights(config, train_ds)
        assert len(class_weights) == nclasses, "Class weights should be of length nclasses."
        train_stats, start_epoch = set_up_resume(config, model, optimizer)
        not_improved = 0
        
        for factor in ["v1", "v2"]:
            config.update({"factor": factor})
            criterion, mus, mu = set_up_criterion(config, class_weights, nclasses)
            assert len(mus) == nclasses, "Mus should be of length nclasses."
            assert criterion.__class__.__name__ == "DailyRewardPiecewiseLinRegrLoss", "Criterion should be a DailyRewardPiecewiseLinRegrLoss."
            assert criterion.factor == factor, "Factor should be the same."
        
            with tqdm(range(start_epoch, config.epochs + 1)) as pbar:
                for epoch in pbar:
                    if mus_should_be_updated(config, epoch):
                        mus = update_mus_during_training(config, criterion, stats, epoch, mus, mu)
                    
                    # train and test epoch
                    dict_args = {"epoch": epoch}
                    trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=config.device, extra_padding_list=extra_padding_list, **dict_args)
                    testloss, stats = run_test_epoch(model, testdataloader, criterion, config, extra_padding_list=extra_padding_list, return_id=test_ds.return_id, **dict_args)
                    assert isinstance(trainloss, (float, np.float32, np.float64)), "Train loss should be a float."
                    assert isinstance(testloss, (float, np.float32, np.float64)), "Test loss should be a float."
                    assert isinstance(stats, dict), "Stats should be a dictionary."
                    assert "wrong_pred_penalty" in stats, "Stats should contain wrong_pred_penalty."
                    assert isinstance(stats["wrong_pred_penalty"], (np.ndarray, torch.Tensor)), "Wrong pred penalty should be an array."
                    assert sum(criterion.alphas) == 1, "Alphas should sum to 1."
                    break
            

            
            
            