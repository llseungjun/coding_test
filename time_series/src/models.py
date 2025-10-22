import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=1, out=1, drop=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=drop if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, out)

    def forward(self, x):
        o, _ = self.lstm(x)  # (B, W, H)
        h_last = o[:, -1, :]
        return self.fc(h_last)
