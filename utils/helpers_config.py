import torch 
import os 
import json
import wandb
from yaml import safe_load 

def load_personal_config(path = os.path.join("..", "config", "personal_config.yaml")):
    """ load the personal config in config/personal_config.yaml"""
    with open(path, 'r') as file:
        config = safe_load(file)
    entity = config["entity"]
    project = config["project"]
    sweep = config["sweep"]
    assert len(entity) > 0, "entity should not be empty, please fill in the personal_config.yaml file"
    assert len(project) > 0, "project should not be empty, please fill in the personal_config.yaml file"
    if len(sweep) == 0:
        # if sweep is empty, only give a warning. 
        print("sweep is empty in personal_config.yaml file.")
    return entity, project, sweep

def set_up_config(config, print_comments:bool=False, final_train:bool=False):
    """ sets up the configuration for training and testing. If the configuration is not set, it sets the default values.
    
    Args:
        config (argparse.Namespace): configuration
        print_comments (bool): whether to print comments or not
    
    Returns:
        config (argparse.Namespace): configuration
    """
    # Check if config is an instance of wandb Config
    if isinstance(config, wandb.Config):
        update_dict = {}
    if not hasattr(config, "device"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(config, wandb.Config):
            update_dict['device'] = device
        else:
            config.device = device
        if print_comments:
            print(f"device set to {config.device}")
    if not hasattr(config, "backbonemodel"):
        backbonemodel = "LSTM"
        if isinstance(config, wandb.Config):
            update_dict['backbonemodel'] = backbonemodel
        else:
            config.backbonemodel = backbonemodel
        if print_comments:
            print(f"backbone model set to {config.backbonemodel}")
    if not hasattr(config, "hidden_dims"):
        hidden_dims = 64
        if isinstance(config, wandb.Config):
            update_dict['hidden_dims'] = hidden_dims
        else:
            config.hidden_dims = hidden_dims
        if print_comments:
            print(f"hidden_dims set to {config.hidden_dims}")
    if not hasattr(config, "decision_head"):
        decision_head = ""
        if isinstance(config, wandb.Config):
            update_dict['decision_head'] = decision_head
        else:
            config.decision_head = decision_head
        if print_comments:
            print(f"decision_head set to {config.decision_head}")
    if not hasattr(config, "loss"):
        loss = "early_reward"
        if isinstance(config, wandb.Config):
            update_dict['loss'] = loss
        else:
            config.loss = loss
        if print_comments:
            print(f"loss set to {config.loss}")
    if not hasattr(config, "validation_set"):
        if final_train:
            validation_set = "eval"
        else: 
            validation_set = "valid"
        if isinstance(config, wandb.Config):
            update_dict['validation_set'] = validation_set
        else:
            config.validation_set = validation_set
        if print_comments:
            print(f"validation_set set to {config.validation_set}")
    else: 
        if final_train and config.validation_set != "eval":
            if isinstance(config, wandb.Config):
                update_dict['validation_set'] = 'eval'
            else:
                config.validation_set = 'eval'
            if print_comments:
                print(f"validation_set set to {config.validation_set}")
    if not hasattr(config, "daily_timestamps"):
        daily_timestamps = False
        if isinstance(config, wandb.Config):
            update_dict['daily_timestamps'] = daily_timestamps
        else:
            config.daily_timestamps = daily_timestamps
        if print_comments:
            print(f"daily_timestamps set to {config.daily_timestamps}")
    if not hasattr(config, "alpha_decay"):
        alpha_decay = [config.alpha, config.alpha]
        if isinstance(config, wandb.Config):
            update_dict['alpha_decay'] = alpha_decay
        else:
            config.alpha_decay = alpha_decay
        if print_comments:
            print(f"alpha_decay set to {config.alpha_decay}")
    if not hasattr(config, "corrected"):
        corrected = False
        if isinstance(config, wandb.Config):
            update_dict['corrected'] = corrected
        else:
            config.corrected = corrected
        if print_comments:
            print(f"corrected set to {config.corrected}")
    if not hasattr(config, "original_time_serie_lengths"):
        original_time_serie_lengths = None
        if isinstance(config, wandb.Config):
            update_dict['original_time_serie_lengths'] = original_time_serie_lengths
        else:
            config.original_time_serie_lengths = original_time_serie_lengths
        if print_comments:
            print(f"original_time_serie_lengths set to {config.original_time_serie_lengths}")
    if hasattr(config, "class_weights"):
        if isinstance(config.class_weights, list):  # Check if it's a list
            class_weights = torch.tensor(config.class_weights, dtype=torch.float)  
        else: 
            class_weights = config.class_weights
        class_weights = class_weights.clone().detach().to(config.device)
        if isinstance(config, wandb.Config):
            update_dict['class_weights'] = class_weights
        else:
            config.class_weights = class_weights
        if print_comments:
            print(f"weights moved to device {config.device}")
            
    # If config is a wandb Config and there are updates, apply them
    if isinstance(config, wandb.Config) and update_dict:
        config.update(update_dict, allow_val_change=True)
    
    return config


def save_config(model_path: str, run):
    """ save the run configuration as a json file in model_path/config.json
    
    Args:
        model_path (str): path to the model directory
        run (argparse.Namespace): run configuration
    
    Returns:
        str: path to the saved config file
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(run.config, f)
    print("config file saved at: ", config_path)
    return config_path

def print_config(run):
    print("-"*50, "Configuration:", "-"*50)
    for key, value in run.config.items():
        print(f"{key}: {value}")
    print("-"*150)