import torch
import torch.nn as nn

class ClassificationHead(torch.nn.Module):

    def __init__(self, hidden_dims, nclasses):
        super(ClassificationHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, nclasses, bias=True),
            nn.LogSoftmax(dim=2))

    def forward(self, x):
        return self.projection(x)

class DecisionHead(torch.nn.Module):

    def __init__(self, hidden_dims):
        super(DecisionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, 1, bias=True),
            nn.Sigmoid()
        )

        # initialize bias to predict late in first epochs
        torch.nn.init.normal_(self.projection[0].bias, mean=-2e1, std=1e-1)

    def forward(self, x):
        return self.projection(x).squeeze(2)
    

class DecisionHeadDay(torch.nn.Module):
    
    def __init__(self, hidden_dims) -> None:
        super(DecisionHeadDay, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, 1, bias=True),
            nn.Sigmoid()
        )
        # initialize bias to predict late in first epochs
        torch.nn.init.normal_(self.projection[0].bias, mean=15, std=5)

    def forward(self, x):
        x = self.projection(x).squeeze(2)
        x = x*364
        return x