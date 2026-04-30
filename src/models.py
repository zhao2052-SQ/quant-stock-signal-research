import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_sklearn_model(name):
    if name == "logistic":
        return LogisticRegression(max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    raise ValueError(f"Unknown sklearn model: {name}")

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)

class CNN2DClassifier(nn.Module):
    def __init__(self, lookback, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * lookback * input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)
