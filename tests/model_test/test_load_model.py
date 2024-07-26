import sys
import os
import wandb
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.test.load_model import (
    get_all_runs,
    get_best_run,
    download_model,
    get_model_and_model_path,
    get_loaded_model_and_criterion,
)
from utils.helpers_config import load_personal_wandb_config


class TestLoadModel:
    def test_get_all_runs(self):
        entity, project, _ = load_personal_wandb_config(
            os.path.join(parent_dir, "config", "personal_config.yaml")
        )
        runs_df, runs = get_all_runs(entity, project)
        assert len(runs_df) > 0
        assert len(runs_df) <= len(runs)
        assert runs_df.columns.tolist() == [
            "summary",
            "config",
            "name",
            "sweep",
            "start_date",
        ]
        assert all([isinstance(run, wandb.apis.public.Run) for run in runs])
        assert all([isinstance(name, str) for name in runs_df.name])
        assert all([isinstance(config, dict) for config in runs_df.config])
        assert all([isinstance(name, str) for name in runs_df.name])
        assert all([isinstance(sweep, str) or sweep is None for sweep in runs_df.sweep])
        assert all([isinstance(start_date, str) for start_date in runs_df.start_date])
        assert all(
            [
                config.get("backbonemodel", None) != "TempCNN"
                for config in runs_df.config
            ]
        )

    def test_get_best_run(self):
        entity, project, _ = load_personal_wandb_config(
            os.path.join(parent_dir, "config", "personal_config.yaml")
        )
        runs_df, runs = get_all_runs(entity, project)
        metric = "harmonic_mean"
        chosen_run = get_best_run(runs_df, runs, metric)
        assert isinstance(chosen_run, wandb.apis.public.runs.Run)
        assert isinstance(chosen_run.name, str)
        assert isinstance(chosen_run.summary._json_dict[metric], float)
        # check that indeed the chosen run has the highest value of the metric
        runs_lstm = [
            run for run in runs if run.config.get("backbonemodel", None) != "TempCNN"
        ]
        max_metric_value = max(
            [run.summary._json_dict.get(metric, float("-inf")) for run in runs_lstm]
        )
        if max_metric_value != float(
            "-inf"
        ):  # Ensure there is at least one valid metric value
            assert (
                chosen_run.summary._json_dict.get(metric, float("-inf"))
                == max_metric_value
            )
        else:
            raise ValueError("No valid metric values found for 'harmonic_mean'.")

    def test_download_model(self):
        entity, project, _ = load_personal_wandb_config(
            os.path.join(parent_dir, "config", "personal_config.yaml")
        )
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
        entity, project, _ = load_personal_wandb_config(
            os.path.join(parent_dir, "config", "personal_config.yaml")
        )
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
        entity, project, _ = load_personal_wandb_config(
            os.path.join(parent_dir, "config", "personal_config.yaml")
        )
        runs_df, runs = get_all_runs(entity, project)
        metric = "harmonic_mean"
        chosen_run = get_best_run(runs_df, runs, metric)
        nclasses, input_dim = 9, 13
        mus = [150.0] * nclasses
        model, criterion = get_loaded_model_and_criterion(
            chosen_run, nclasses, input_dim, mus=mus
        )
        assert isinstance(model, torch.nn.Module)
        assert isinstance(criterion, torch.nn.Module)
