import torch
import torch.nn as nn


def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_backbone_model(
    backbone_model,
    input_dim,
    hidden_dims,
    nclasses,
    num_rnn_layers,
    dropout,
    sequencelength,
    kernel_size,
):
    model_map = {
        "LSTM": {
            "class": nn.LSTM,
            "config": {
                "input_size": hidden_dims,
                "hidden_size": hidden_dims,
                "num_layers": num_rnn_layers,
                "bias": False,
                "batch_first": True,
                "dropout": dropout,
                "bidirectional": False,
            },
        },
    }

    if backbone_model in model_map:
        model_info = model_map[backbone_model]
        return model_info["class"](**model_info["config"])
    else:
        raise ValueError(f"Backbone model {backbone_model} is not implemented yet.")


def get_t_stop_from_daily_timestamps(timestamps_left, threshold=1):
    # t_stop is the time at which the model stops. It is the first time the timestamps_left is strictly smaller than threshold
    batchsize, sequencelength = timestamps_left.shape
    time_smaller_than_1 = (timestamps_left < threshold).int()
    t_stop = torch.argmax(time_smaller_than_1, dim=1)  # shape: (batchsize,)
    # for t_stop==0 and time_smaller_than_1==0, set t_stop to sequencelength-1 (for the case where timestamps_left is always larger than threshold)
    idx = torch.where((t_stop == 0) & (time_smaller_than_1[:, 0] == 0))
    t_stop[idx] = sequencelength - 1

    return t_stop
