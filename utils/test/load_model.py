import wandb
import pandas as pd
import json
import os
import argparse
import torch 
from models.model_helpers import count_parameters
from utils.helpers_training import set_up_criterion, set_up_model
from utils.helpers_config import set_up_config


def get_all_runs(entity, project):
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


def get_best_run(df, runs, metric):
    # check if metric is in summary
    if not all([metric in x for x in df.summary]):
        print(f"metric {metric} not in summary")
        print("available metrics: ", df.summary[0].keys())
        return None
    chosen_run_idx = df.summary.apply(lambda x: x[metric]).idxmax()
    chosen_run = runs[chosen_run_idx]
    print("chosen run: ", chosen_run.name)
    print(f"with {metric}: ", chosen_run.summary._json_dict[metric])
    return chosen_run


def download_model(chosen_run):
    artifacts = chosen_run.logged_artifacts()
    model = [artifact for artifact in artifacts if artifact.type == "model"][-1] # get the latest model artifact
    model_path = model.download()
    print("model path:", model_path)

    # save config as json file
    with open(os.path.join(model_path,"args.json"), "w") as f:
        json.dump(chosen_run.config, f)
        
        
def get_model_and_model_path(run):
    artifacts = run.logged_artifacts()
    model = [artifact for artifact in artifacts if artifact.type == "model"][-1] # get the latest model artifact
    model_path = model.download()
    return model, model_path


def get_loaded_model_and_criterion(run, nclasses, input_dim, mus=None):
    model, model_path = get_model_and_model_path(run)
    run_config = argparse.Namespace(**run.config)
    run_config, _ = set_up_config(run_config)
    if hasattr(run_config, "class_weights") and run_config.class_weights is not None:
        class_weights = torch.tensor(run_config.class_weights)
    else:
        class_weights = None
    criterion, mus, mu = set_up_criterion(run_config, class_weights, nclasses, mus)
    model = set_up_model(run_config, nclasses, input_dim, train=False)
    print("model is loading from: ", model_path)
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
    print(f"The model has {count_parameters(model):,} trainable parameters.")
    model = model.to(run_config.device)
    criterion = criterion.to(run_config.device)
    model.eval()
    return model, criterion