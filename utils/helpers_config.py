import torch 

def set_up_config(config, print_comments:bool=False):
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
    if not hasattr(config, "decision_head"):
        config.decision_head = ""
    if not hasattr(config, "loss"):
        config.loss = "early_reward"
    if not hasattr(config, "validation_set"):
        config.validation_set = "valid"
    if not hasattr(config, "left_padding"):
        config.left_padding = False
    if not hasattr(config, "extra_padding_list"):
        config.extra_padding_list = [0]
    if not hasattr(config, "daily_timestamps"):
        config.daily_timestamps = False
    if not hasattr(config, "alpha_decay"):
        config.alpha_decay = [config.alpha, config.alpha]
    # only use extra padding if tempcnn
    if config.backbonemodel == "LSTM":
        extra_padding_list = [0]
        if print_comments:
            print(f"Since LSTM is used, extra padding is set to {extra_padding_list}")
    else:
        extra_padding_list = config.extra_padding_list
    
    return config, extra_padding_list