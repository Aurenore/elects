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
    if not hasattr(config, "left_padding"):
        config.left_padding = False
        if print_comments:
            print(f"left_padding set to {config.left_padding}")
    if not hasattr(config, "extra_padding_list"):
        config.extra_padding_list = [0]
        if print_comments:
            print(f"extra_padding_list set to {config.extra_padding_list}")
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
    # only use extra padding if tempcnn
    if config.backbonemodel == "LSTM":
        extra_padding_list = [0]
        if print_comments:
            print(f"Since LSTM is used, extra padding is set to {extra_padding_list}")
    else:
        extra_padding_list = config.extra_padding_list
        
    if hasattr(config, "class_weights"):
        config.class_weights = config.class_weights.clone().detach().to(config.device)
        if print_comments:
            print(f"weights moved to device {config.device}")
    
    return config, extra_padding_list