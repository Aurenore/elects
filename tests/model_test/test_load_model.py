import sys 
import os 
import numpy as np
import wandb
import torch
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.test.load_model import get_all_runs, get_best_run, download_model, get_model_and_model_path, get_loaded_model_and_criterion

class TestLoadModel(): 
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
    
    def test_get_all_runs(self):
        entity, project = "aurenore", "MasterThesis"
        runs_df, runs = get_all_runs(entity, project)
        assert len(runs_df) > 0 
        assert len(runs_df) == len(runs)
        assert runs_df.columns.tolist() == ["summary", "config", "name", "sweep", "start_date"]
        assert all([isinstance(run, wandb.apis.public.Run) for run in runs])
        assert all([isinstance(name, str) for name in runs_df.name])
        assert all([isinstance(config, dict) for config in runs_df.config])
        assert all([isinstance(name, str) for name in runs_df.name])
        assert all([isinstance(sweep, str) or sweep is None for sweep in runs_df.sweep])
        assert all([isinstance(start_date, str) for start_date in runs_df.start_date])
        
    def test_get_best_run(self):
        entity, project = "aurenore", "MasterThesis"
        runs_df, runs = get_all_runs(entity, project)
        metric = "harmonic_mean"
        chosen_run = get_best_run(runs_df, runs, metric)
        assert isinstance(chosen_run, wandb.apis.public.runs.Run)
        assert isinstance(chosen_run.name, str)
        assert isinstance(chosen_run.summary._json_dict[metric], float)
        # check that indeed the chosen run has the highest value of the metric
        max_metric_value = max([run.summary._json_dict.get(metric, float('-inf')) for run in runs])
        if max_metric_value != float('-inf'):  # Ensure there is at least one valid metric value
            assert chosen_run.summary._json_dict.get(metric, float('-inf')) == max_metric_value
        else:
            raise ValueError("No valid metric values found for 'harmonic_mean'.")
            
    def test_download_model(self):
        entity, project = "aurenore", "MasterThesis"
        runs_df, runs = get_all_runs(entity, project)
        metric = "harmonic_mean"
        chosen_run = get_best_run(runs_df, runs, metric)
        model_path = download_model(chosen_run)
        assert isinstance(model_path, str)
        assert os.path.exists(model_path)
        assert os.path.isdir(model_path)
        assert os.path.exists(os.path.join(model_path, "args.json"))
        assert os.path.isfile(os.path.join(model_path, "args.json"))
        assert os.path.getsize(os.path.join(model_path, "args.json")) > 0 
        
    def test_get_model_and_model_path(self):
        entity, project = "aurenore", "MasterThesis"
        runs_df, runs = get_all_runs(entity, project)
        metric = "harmonic_mean"
        chosen_run = get_best_run(runs_df, runs, metric)
        model_artifact, model_path = get_model_and_model_path(chosen_run)
        assert isinstance(model_artifact, wandb.Artifact)
        assert isinstance(model_path, str)
        assert os.path.exists(model_path)
        assert os.path.isdir(model_path)
        assert os.path.exists(os.path.join(model_path, "model.pth"))
        
    def test_get_loaded_model_and_criterion(self):
        entity, project = "aurenore", "MasterThesis"
        runs_df, runs = get_all_runs(entity, project)
        metric = "harmonic_mean"
        chosen_run = get_best_run(runs_df, runs, metric)
        nclasses, input_dim = 9, 13
        mus = [150.]*nclasses
        model, criterion = get_loaded_model_and_criterion(chosen_run, nclasses, input_dim, mus=mus)
        assert isinstance(model, torch.nn.Module)
        assert isinstance(criterion, torch.nn.Module)
        

        
        
        

                    
    

