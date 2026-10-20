"""PyTorch experimental models; manual computation lives in nn_numpy separately."""

from typing import cast

import torch
from torch import nn

from neuropulse.config import TrainConfig


class WindowModel(nn.Module):
    def __init__(self, features: int, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.features = features
        activation = nn.ReLU if config.activation == "relu" else nn.Tanh
        family, hidden = config.family, config.hidden_size
        if family in ("mlp", "autoencoder"):
            size = features * config.window_length
            if family == "autoencoder":
                self.network = nn.Sequential(
                    nn.Linear(size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(32, 128),
                    nn.ReLU(),
                    nn.Linear(128, size),
                )
            else:
                layers: list[nn.Module] = [nn.Flatten()]
                previous = size
                for depth in range(config.hidden_layers):
                    width = max(8, config.dense_width // (2**depth))
                    layers.extend(
                        [nn.Linear(previous, width), activation(), nn.Dropout(config.dropout)]
                    )
                    previous = width
                layers.append(nn.Linear(previous, 1))
                self.network = nn.Sequential(*layers)
        elif family in ("cnn", "cnn_lstm"):
            self.convolution = nn.Sequential(nn.Conv1d(features, 32, 5, padding=2), nn.ReLU())
            if family == "cnn":
                self.convolution.append(nn.Conv1d(32, 64, 3, padding=1))
                self.convolution.append(nn.ReLU())
                hidden = 64
            else:
                self.recurrent = nn.LSTM(32, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), activation(), nn.Dropout(config.dropout), nn.Linear(32, 1)
            )
        elif family == "lstm":
            self.recurrent = nn.LSTM(features, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), activation(), nn.Dropout(config.dropout), nn.Linear(32, 1)
            )
        else:
            raise ValueError("Not a PyTorch family")
        self.initialize()

    def initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                mode = self.config.initialization
                is_output = isinstance(module, nn.Linear) and module.out_features == 1
                if mode == "he" or (
                    mode == "appropriate" and self.config.activation == "relu" and not is_output
                ):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.LSTM):
                for name, parameter in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(parameter)
                    else:
                        nn.init.zeros_(parameter)
                        if "bias_ih" in name:
                            with torch.no_grad():
                                parameter[module.hidden_size : 2 * module.hidden_size].fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.config.window_length, self.features):
            raise ValueError("Expected [B,W,F] matching architecture")
        family = self.config.family
        if family == "mlp":
            return cast(torch.Tensor, self.network(x))
        if family == "autoencoder":
            return cast(torch.Tensor, self.network(x.flatten(1)).reshape_as(x))
        if family in ("cnn", "cnn_lstm"):
            representation = self.convolution(x.transpose(1, 2))
            if family == "cnn":
                return cast(torch.Tensor, self.head(representation.mean(dim=2)))
            x = representation.transpose(1, 2)
        sequence, _ = self.recurrent(x)  # Zero state per independent window, including replay.
        return cast(torch.Tensor, self.head(sequence[:, -1]))
