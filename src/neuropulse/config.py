"""Validated scientific configuration; transformations are implemented in Phase 2."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """The frozen FD001 protocol defaults, with unknown keys rejected.

    No configuration load acquires data or fits preprocessing. YAML is safe-loaded
    and scalar types are strict so strings and booleans cannot silently become seeds.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    dataset: Literal["cmapss"] = "cmapss"
    subset: Literal["FD001"] = "FD001"
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    split_seed: int = Field(default=42, ge=0)
    validation_prefix_seed: int = Field(default=142, ge=0)
    window_length: int = Field(default=30, gt=0, le=100)
    stride: int = Field(default=1, gt=0)
    rul_cap_cycles: int | None = Field(default=125, gt=0)
    sensor_variance_threshold: float = Field(default=1e-8, ge=0.0, allow_inf_nan=False)
    scaler_std_floor: float = Field(default=1e-12, gt=0.0, allow_inf_nan=False)


def load_data_config(path: Path) -> DataConfig:
    """Load an explicit YAML path, failing on non-mappings or invalid values."""
    with path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)
    if not isinstance(document, dict):
        raise ValueError("Data configuration must be a YAML mapping")
    return DataConfig.model_validate(document)


class TrainConfig(BaseModel):
    """Explicit resolved run budget and architecture; validation-only selection."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    family: Literal[
        "linear", "random_forest", "numpy_mlp", "mlp", "cnn", "lstm", "cnn_lstm", "autoencoder"
    ] = "lstm"
    seed: int = Field(default=42, ge=0)
    window_length: int = Field(default=30, gt=0, le=100)
    rul_cap_cycles: int | None = Field(default=125, gt=0)
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=64, gt=0)
    patience: int = Field(default=10, gt=0)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    optimizer: Literal["adam", "sgd", "momentum"] = "adam"
    hidden_size: int = Field(default=64, gt=0)
    dense_width: int = Field(default=128, gt=0)
    hidden_layers: int = Field(default=2, gt=0, le=3)
    activation: Literal["relu", "tanh"] = "relu"
    initialization: Literal["appropriate", "he", "xavier"] = "appropriate"
    dropout: float = Field(default=0.1, ge=0, lt=1, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0, allow_inf_nan=False)


def load_train_config(path: Path) -> TrainConfig:
    with path.open(encoding="utf-8") as config_file:
        return TrainConfig.model_validate(yaml.safe_load(config_file))
