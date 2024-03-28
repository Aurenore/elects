import os
import torch
import torch.nn as nn
import torch.utils.data

"""
Source: https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TempCNN.py 

Pytorch re-implementation of Pelletier et al. 2019
https://github.com/charlotte-pel/temporalCNN

https://www.mdpi.com/2072-4292/11/5/523

We remove the flatten layer in the original implementation, such that we still have the time stamps dimensions.
The output of the model is (batch_size, sequencelength, hidden_dims) instead of (batch_size, num_classes).

"""

__all__ = ['TempCNN']

class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=13, sequencelength=70, kernel_size=7, hidden_dims=64, dropout=0.18203942949809093):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_sequencelenght={sequencelength}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout) 
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims, 4*hidden_dims, drop_probability=dropout) 
        self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, hidden_dims), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        # require NxTxD (batch_size, sequencelength, input_dim)
        x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = x.transpose(1,2) # (batch_size, sequencelength, hidden_dims)
        #x = self.flatten(x) remove it such that we still have the time stamps dimensions. 
        x = self.dense(x) # (batch_size, 4*hidden_dims, sequencelength)
        x = x.transpose(1,2) # (batch_size, 4*hidden_dims, sequencelength) -> (batch_size, sequencelength, 4*hidden_dims)
        x = self.logsoftmax(x) # (batch_size, sequencelength, hidden_dims)
        return x

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()
        self.lin = nn.Linear(input_dim, hidden_dims) # (batch_size, sequencelength, input_dim) -> (batch_size, sequencelength, hidden_dims)
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dims), # (batch_size, hidden_dims, sequencelength) -> (batch_size, hidden_dims, sequencelength)
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        X = self.lin(X)
        X = X.transpose(1,2)
        X = self.block(X)
        return X
