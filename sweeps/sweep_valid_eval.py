import wandb
wandb.login()
import yaml
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
DATA = cfg["DATA"]
PROJECTUSER_PATH = cfg["PROJECTUSER_PATH"]
import torch

sweep_configuration = {
    "method": "random",
    "name": "sweep_valid_eval",
    "metric": {"goal": "maximize", "name": "harmonic_mean"},
    "parameters": {
        "backbonemodel": {"values": ["TempCNN", "LSTM"]},
        "dataset": {"value": DATA},
        "alpha": {"value": 0.5}, 
        "epsilon": {"value": 10},
        "learning_rate": {"value": 1e-3},
        "weight_decay": {"value": 0},
        "patience": {"value": 30},
        "device": {"value": "cuda" if torch.cuda.is_available() else "cpu"},
        "epochs": {"value": 100},
        "sequencelength": {"value": 70},
        "extra_padding_list": {"value": [50, 40, 30, 20, 10, 0]},
        "hidden_dims": {"value": 64}, 
        "batchsize": {"value": 256},
        "dataroot": {"value": f"{PROJECTUSER_PATH}/elects_data"},
        "snapshot": {"value": f"{PROJECTUSER_PATH}/elects_snapshots/{DATA}/model.pth"},
        "left_padding": {"value": False}, 
        "loss_weight": {"value": None}, 
        "resume": {"value": False},
        "validation_set": {"values": ["valid", "eval"]}
    },
}

wandb.sweep(sweep=sweep_configuration, project="MasterThesis")