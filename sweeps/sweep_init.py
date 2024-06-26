import os 
PROJECTUSER_PATH="/mydata/studentanya/anya"
DATA="breizhcrops"

sweep_configuration = {
    "method": "grid",
    "name": "sweep_balanced_left_right_padding_local",
    "metric": {"goal": "maximize", "name": "harmonic_mean"},
    "parameters": {
        "backbonemodel": {"value": "TempCNN"},
        "dataset": {"value": DATA},
        "alpha": {"value": 0.5}, # {"min": 0.1, "max": 0.9, "distribution": "uniform"},
        "epsilon": {"value": 10},
        "learning_rate": {"value": 1e-3},
        "weight_decay": {"value": 0},
        "patience": {"value": 30},
        "device": {"value": "cuda"},
        "epochs": {"value": 100},
        "sequencelength": {"value": 70},
        "extra_padding_list": {"value": [50, 40, 30, 20, 10, 0]},
        "hidden_dims": {"values": [16, 32, 64]}, #{"value": 64}, #
        "batchsize": {"value": 256},
        "dataroot": {"value": os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data")}, # {"value": f"{PROJECTUSER_PATH}/elects_data"},
        "snapshot": {"value": os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data", "snapshots", "model.pth")}, #{"value": f"{PROJECTUSER_PATH}/elects_snapshots/{DATA}/model.pth"},
        "left_padding": {"values": [True, False]}, # {"value": False}, # 
        "loss_weight": {"values": [None, "balanced"]}, # {"value": "balanced"}, #
        "resume": {"value": False},
        "validation_set": {"value": "valid"}, #{"values": ["valid", "eval"]}
    },
}