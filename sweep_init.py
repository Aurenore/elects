import wandb
wandb.login()

PROJECTUSER_PATH="/mydata/studentanya/anya"
DATA="breizhcrops"

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "harmonic_mean"},
    "parameters": {
        "backbonemodel": {"value": "TempCNN"},
        "dataset": {"value": DATA},
        "alpha": {"value": 0.5},
        "epsilon": {"value": 10},
        "learning_rate": {"value": 1e-3},
        "weight_decay": {"value": 0},
        "patience": {"value": 30},
        "device": {"value": "cuda"},
        "epochs": {"value": 100},
        "sequence_length": {"value": 70},
        "extra_padding_list": {"value": [50, 40, 30, 20, 10, 0]},
        "hidden_dims": {"values": [16, 32, 64]},
        "batchsize": {"value": 256},
        "dataroot": {"value": f"{PROJECTUSER_PATH}/elects_data"},
        "snapshot": {"value": f"{PROJECTUSER_PATH}/elects_snapshots/{DATA}/model.pth"},
        "left_padding": {"values": [True, False]},
        "resume": {"value": False},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="MasterThesis")
print("sweep id: ", sweep_id)