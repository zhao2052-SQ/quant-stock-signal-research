import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from .models import build_sklearn_model, LSTMClassifier, CNN2DClassifier

def train_sklearn_model(df, columns, train_ratio, model_name):
    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[columns])
    x_test = scaler.transform(test_df[columns])
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    model = build_sklearn_model(model_name)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    return pred, test_df["date"].values, test_df["future_return"].values, metrics

def train_torch_sequence_model(x, y, train_ratio, model_name, epochs=20, lr=0.001):
    split = int(len(x) * train_ratio)
    x_train = torch.tensor(x[:split])
    y_train = torch.tensor(y[:split])
    x_test = torch.tensor(x[split:])
    y_test = torch.tensor(y[split:])
    input_dim = x.shape[-1]
    lookback = x.shape[1]
    if model_name == "lstm":
        model = LSTMClassifier(input_dim)
    elif model_name == "cnn2d":
        model = CNN2DClassifier(lookback, input_dim)
    else:
        raise ValueError(f"Unknown torch model: {model_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        pred = logits.argmax(dim=1).numpy()
    metrics = {
        "accuracy": float(accuracy_score(y_test.numpy(), pred)),
        "f1": float(f1_score(y_test.numpy(), pred, zero_division=0)),
    }
    return pred, split, metrics
