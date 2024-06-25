import torch 
import os 
import json

def set_up_config(config, print_comments:bool=False):
    """ sets up the configuration for training and testing. If the configuration is not set, it sets the default values.
    
    Args:
        config (argparse.Namespace): configuration
        print_comments (bool): whether to print comments or not
    
    Returns:
        config (argparse.Namespace): configuration
    """
    if not hasattr(config, "device"):
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if print_comments:
            print(f"device set to {config.device}")
    if not hasattr(config, "backbonemodel"):
        config.backbonemodel = "LSTM"
        if print_comments:
            print(f"backbone model set to {config.backbonemodel}")
    if not hasattr(config, "hidden_dims"):
        config.hidden_dims = 64
        if print_comments:
            print(f"hidden_dims set to {config.hidden_dims}")
    if not hasattr(config, "decision_head"):
        config.decision_head = ""
        if print_comments:
            print(f"decision_head set to {config.decision_head}")
    if not hasattr(config, "loss"):
        config.loss = "early_reward"
        if print_comments:
            print(f"loss set to {config.loss}")
    if not hasattr(config, "validation_set"):
        config.validation_set = "valid"
        if print_comments:
            print(f"validation_set set to {config.validation_set}")
    if not hasattr(config, "daily_timestamps"):
        config.daily_timestamps = False
        if print_comments:
            print(f"daily_timestamps set to {config.daily_timestamps}")
    if not hasattr(config, "alpha_decay"):
        config.alpha_decay = [config.alpha, config.alpha]
        if print_comments:
            print(f"alpha_decay set to {config.alpha_decay}")
    if not hasattr(config, "corrected"):
        config.corrected = False
        if print_comments:
            print(f"corrected set to {config.corrected}")
    if not hasattr(config, "original_time_serie_lengths"):
        config.original_time_serie_lengths = None
        if print_comments:
            print(f"original_time_serie_lengths set to {config.original_time_serie_lengths}")
        
    if hasattr(config, "class_weights"):
        if isinstance(config.class_weights, list):  # Check if it's a list
            config.class_weights = torch.tensor(config.class_weights, dtype=torch.float)
        config.class_weights = config.class_weights.clone().detach().to(config.device)
        if print_comments:
            print(f"weights moved to device {config.device}")
    
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