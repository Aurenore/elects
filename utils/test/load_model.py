import wandb
import pandas as pd
import json
import os
import argparse
import torch 
from typing import Tuple
from models.model_helpers import count_parameters
from utils.helpers_training import set_up_criterion, set_up_model
from utils.helpers_config import set_up_config


def get_all_runs(entity: str, project: str) -> Tuple[pd.DataFrame, list]:
    """ Get all the runs of a project from wandb.
    
    Args:
        entity (str): the entity
        project (str): the project
    
    Returns:
        runs_df (pd.DataFrame): dataframe with the summary, config, name, sweep, start_date of each run
        runs (list): list of runs
    
    """
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    summary_list, config_list, name_list, sweep_list, start_date_list = [], [], [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        
        if run.sweep:
            sweep_list.append(run.sweep.name)
        else:
            sweep_list.append(None)
            
        start_date_list.append(run.createdAt)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list, "sweep": sweep_list, "start_date": start_date_list}
    )
    return runs_df, runs


def select_rows_with_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Check if the metric is present in the summary of each row
    mask = df.apply(lambda row: metric in row.summary, axis=1)
    
    # Select only the rows where the metric is present
    filtered_df = df[mask]
    
    return filtered_df


def get_best_run(df: pd.DataFrame, runs: list, metric: str) -> wandb.apis.public.runs.Run:
    """ Get the run with the best metric.
    
    Args:
        df (pd.DataFrame): dataframe with the summary, config, name, sweep, start_date of each run
        runs (list): list of runs
        metric (str): the metric to maximize
        
    Returns:
        chosen_run: the run with the best metric
    
    """
    df = select_rows_with_metric(df, metric)
    if len(df) == 0:
        raise ValueError(f"metric {metric} not found in the summary of the runs.")
    chosen_run_idx = df.summary.apply(lambda x: x[metric]).idxmax()
    chosen_run = runs[chosen_run_idx]
    print("chosen run: ", chosen_run.name)
    print(f"with {metric}: ", chosen_run.summary._json_dict[metric])
    return chosen_run


def download_model(chosen_run: wandb.apis.public.runs.Run) -> str:
    """ Download the model from wandb.
    
    Args:
        chosen_run: the run with the model to download
        
    Returns: 
        None
    
    """
    artifacts = chosen_run.logged_artifacts()
    model = [artifact for artifact in artifacts if artifact.type == "model"][-1] # get the latest model artifact
    model_path = model.download()
    print("model path:", model_path)

    # save config as json file
    with open(os.path.join(model_path,"args.json"), "w") as f:
        json.dump(chosen_run.config, f)
        
    return model_path
        
        
def get_model_and_model_path(run: wandb.apis.public.runs.Run) -> Tuple[wandb.Artifact, str]:
    """ get the model artifact and the path to the model. 
        Download the model in the format .pth at model_path.
    
    Args:
        run: the run
        
    Returns:
        model: the model artifact
        model_path: the path to the model
    
    """
    artifacts = run.logged_artifacts()
    model_artifact = [artifact for artifact in artifacts if artifact.type == "model"][-1] # get the latest model artifact
    model_path = model_artifact.download()
    return model_artifact, model_path


def get_loaded_model_and_criterion(run: wandb.apis.public.runs.Run, nclasses: int, input_dim: int, mus: list=None) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """ Load the model and the criterion from a run.
    
    Args:
        run: the run
        nclasses (int): number of classes
        input_dim (int): input dimension
        mus (list): list of mus for each class
    
    Returns:
        model: the model
        criterion: the criterion
        
    """
    model_artifact, model_path = get_model_and_model_path(run)
    run_config = argparse.Namespace(**run.config)
    run_config, _ = set_up_config(run_config)
    if hasattr(run_config, "class_weights") and run_config.class_weights is not None:
        class_weights = torch.tensor(run_config.class_weights)
    else:
        class_weights = None
    criterion, mus, mu = set_up_criterion(run_config, class_weights, nclasses, mus)
    model = set_up_model(run_config, nclasses, input_dim, update_wandb=False)
    print("model is loading from: ", model_path)
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
    print(f"The model has {count_parameters(model):,} trainable parameters.")
    model = model.to(run_config.device)
    criterion = criterion.to(run_config.device)
    model.eval()
    return model, criterion