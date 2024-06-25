import sys
import os 
from tqdm import tqdm
import torch
import numpy as np 
import pytest
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.helpers_config import set_up_config
from utils.helpers_mu import mus_should_be_updated, update_mus_during_training
from utils.helpers_training import train_epoch, set_up_model, set_up_optimizer, set_up_class_weights, \
    set_up_criterion, set_up_resume, get_all_metrics, update_patience, log_description, \
    plots_during_training, load_dataset, plot_label_distribution_in_training 
from utils.helpers_config import set_up_config
from utils.helpers_testing import test_epoch as run_test_epoch
from utils.losses.daily_reward_piecewise_lin_regr_loss import DailyRewardPiecewiseLinRegrLoss

class TestDailyRewardPiecewiseLinRegrLoss(): 
    class Config():
        def __init__(self):
            self.alpha = 1.
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
            self.alpha_decay = [1., 0.6]
            self.mu = 150.
            self.factor = "v2"
            
        def update(self, dict_args: dict):
            for key, value in dict_args.items():
                setattr(self, key, value)
                
    @pytest.fixture
    def config(self):
        return self.Config()
    
    def test_update_alphas(self, config):
        config = set_up_config(config)
        nclasses, input_dim = 7, 13
        class_weights = torch.tensor(np.random.rand(nclasses))
        criterion, mus, mu = set_up_criterion(config, class_weights, nclasses, wandb_update=False)
        if config.loss == "daily_reward_piecewise_lin_regr":
            assert isinstance(criterion, torch.nn.Module), "Criterion should be a torch module."
            assert isinstance(criterion, DailyRewardPiecewiseLinRegrLoss), "Criterion should be a DailyRewardPiecewiseLinRegrLoss."
            assert criterion.alpha == config.alpha, f"Alpha should be {config.alpha}, but is {criterion.alpha}."
            assert criterion.factor == config.factor, f"Factor should be {config.factor}, but is {criterion.factor}."
            assert criterion.alpha_decay_max == config.alpha_decay[0], f"Alpha decay max should be {config.alpha_decay[0]}, but is {criterion.alpha_decay_max}."
            assert criterion.alpha_decay_min == config.alpha_decay[1], f"Alpha decay min should be {config.alpha_decay[1]}, but is {criterion.alpha_decay_min}."
            assert criterion.epochs == config.epochs, f"Epochs should be {config.epochs}, but is {criterion.epochs}."
            assert criterion.start_decision_head_training == config.start_decision_head_training, f"Start decision head training should be {config.start_decision_head_training}, but is {criterion.start_decision_head_training}."
            assert criterion.percentages_other_alphas is not None, "Percentages other alphas should not be None."
        assert len(criterion.percentages_other_alphas) == 3, "Percentages other alphas should have length 3."
        assert torch.isclose(criterion.alphas.sum(), torch.tensor(1.)), f"Alphas should sum to 1, but sum is {criterion.alphas.sum()}."
        
    def test_update_alphas_during_training(self, config):
        config = set_up_config(config)
        nclasses, input_dim = 7, 13
        class_weights = torch.tensor(np.random.rand(nclasses))
        criterion, mus, mu = set_up_criterion(config, class_weights, nclasses, wandb_update=False)
        assert np.isclose(criterion.alphas[0], criterion.alpha), f"{criterion.alphas[0]} != {criterion.alpha}"
        
        for epoch in range(1, 100):
            criterion.update_alphas_at_epoch(epoch, config.device)
            assert np.isclose(sum(criterion.alphas), 1), f"Alphas should sum to 1, but sum is {sum(criterion.alphas)}."
            if epoch >= config.start_decision_head_training:
                assert np.isclose(criterion.alphas[0], criterion.alpha_decay_min + (criterion.alpha_decay_max - criterion.alpha_decay_min) * \
                    (1 - (epoch-config.start_decision_head_training)/(config.epochs-config.start_decision_head_training))), \
                    f"at epoch {epoch}, {criterion.alphas[0]} != {criterion.alpha_decay_min + (criterion.alpha_decay_max - criterion.alpha_decay_min) * (1 - (epoch-config.start_decision_head_training)/(config.epochs-config.start_decision_head_training))}"
                assert np.isclose(criterion.alphas[1], criterion.percentages_other_alphas[0]*(1.-criterion.alpha)), f"at epoch {epoch}, {criterion.alphas[1]} != {criterion.percentages_other_alphas[0]*(1.-criterion.alpha)}" 
                assert np.isclose(criterion.alphas[2], criterion.percentages_other_alphas[1]*(1.-criterion.alpha)), f"at epoch {epoch}, {criterion.alphas[2]} != {criterion.percentages_other_alphas[1]*(1.-criterion.alpha)}"
                assert np.isclose(criterion.alphas[3], criterion.percentages_other_alphas[2]*(1.-criterion.alpha)), f"at epoch {epoch}, {criterion.alphas[3]} != {criterion.percentages_other_alphas[2]*(1.-criterion.alpha)}"         
                    
    def test_instantiation(self, config): 
        torch.autograd.set_detect_anomaly(True)
        config.update({"alpha": 0.9, "alpha_decay": [0.9, 0.6]})
        config = set_up_config(config)
        assert config is not None, "Config setup failed."
        
        traindataloader, testdataloader, train_ds, test_ds, nclasses, class_names, input_dim, length_sorted_doy_dict_test = load_dataset(config)

        model = set_up_model(config, nclasses, input_dim, update_wandb=False)
        optimizer = set_up_optimizer(config, model)
        class_weights = set_up_class_weights(config, train_ds)
        assert len(class_weights) == nclasses, "Class weights should be of length nclasses."
        train_stats, start_epoch = set_up_resume(config, model, optimizer)
        not_improved = 0
        
        for factor in ["v1", "v2"]:
            config.update({"factor": factor})
            criterion, mus, mu = set_up_criterion(config, class_weights, nclasses, wandb_update=False)
            assert len(mus) == nclasses, "Mus should be of length nclasses."
            assert criterion.__class__.__name__ == "DailyRewardPiecewiseLinRegrLoss", "Criterion should be a DailyRewardPiecewiseLinRegrLoss."
            assert criterion.factor == factor, "Factor should be the same."
        
            with tqdm(range(start_epoch, config.epochs + 1)) as pbar:
                for epoch in pbar:
                    if mus_should_be_updated(config, epoch):
                        mus = update_mus_during_training(config, criterion, stats, epoch, mus, mu)
                    
                    # train and test epoch
                    dict_args = {"epoch": epoch}
                    trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=config.device, **dict_args)
                    testloss, stats = run_test_epoch(model, testdataloader, criterion, config, return_id=test_ds.return_id, **dict_args)
                    assert isinstance(trainloss, (float, np.float32, np.float64)), "Train loss should be a float."
                    assert isinstance(testloss, (float, np.float32, np.float64)), "Test loss should be a float."
                    assert isinstance(stats, dict), "Stats should be a dictionary."
                    assert "wrong_pred_penalty" in stats, "Stats should contain wrong_pred_penalty."
                    assert isinstance(stats["wrong_pred_penalty"], (np.ndarray, torch.Tensor)), "Wrong pred penalty should be an array."
                    assert sum(criterion.alphas) == 1, "Alphas should sum to 1."
                    break
                break
        torch.autograd.set_detect_anomaly(False)