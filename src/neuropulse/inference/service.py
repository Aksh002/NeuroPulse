"""Ground-truth-free observation contracts and shared chronological-prefix inference."""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from neuropulse.decisions.rules import DemoRules, RuleState
from neuropulse.inference.bundles import ModelBundle

Finite = Annotated[float, Field(allow_inf_nan=False)]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    dataset: Literal["cmapss"] = "cmapss"
    subset: Literal["FD001"] = "FD001"
    partition: Literal["train", "validation", "test"] = "validation"
    engine_id: int = Field(gt=0)
    cycle: int = Field(gt=0)
    operating_settings: list[Finite] = Field(min_length=3, max_length=3)
    sensors: list[Finite] = Field(min_length=21, max_length=21)
    source: Literal["historical_replay", "historical_upload"] = "historical_upload"


class Prediction(BaseModel):
    schema_version: str = "1.0"
    machine_key: str
    cycle: int
    status: Literal["ready", "warming_up"]
    source: str
    model_version: str
    anomaly_model_version: str | None = None
    rul_cycles: float | None = None
    raw_rul_cycles: float | None = None
    uncertainty: None = None
    anomaly_score: float | None = None
    anomaly_threshold: float | None = None
    health_index: float | None = None
    health_index_kind: str = "rul_derived"
    recommendation: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    rule_version: str = "demo-v1"
    history_length: int
    padding_applied: bool = False


def infer_prefix(
    observations: list[Observation],
    bundle: ModelBundle,
    anomaly: ModelBundle | None = None,
    prior: RuleState | None = None,
    rules: DemoRules | None = None,
) -> tuple[Prediction, RuleState]:
    if not observations:
        raise ValueError("Empty prefix")
    last = observations[-1]
    for previous, current in zip(observations, observations[1:], strict=False):
        if (previous.dataset, previous.subset, previous.partition, previous.engine_id) != (
            current.dataset,
            current.subset,
            current.partition,
            current.engine_id,
        ):
            raise ValueError("Mixed engine identities")
        if current.cycle != previous.cycle + 1:
            raise ValueError("Observations must be consecutive and increasing")
    if bundle.config.family == "autoencoder":
        raise ValueError("RUL branch requires a regressor")
    if anomaly is not None and not bundle.compatible(anomaly):
        raise ValueError("Incompatible anomaly branch")
    key = f"cmapss:FD001:{last.partition}:{last.engine_id}"
    rule_config = rules or DemoRules()
    result: dict[str, Any] = {
        "machine_key": key,
        "cycle": last.cycle,
        "source": last.source,
        "model_version": bundle.manifest["model_version"],
        "history_length": len(observations),
        "anomaly_model_version": anomaly.manifest["model_version"] if anomaly else None,
        "rule_version": rule_config.version,
        "status": "warming_up",
    }
    if len(observations) < bundle.config.window_length:
        return Prediction.model_validate(result), prior or RuleState()
    sensors = np.array([o.sensors for o in observations[-bundle.config.window_length :]])
    window = bundle.scaler.transform(sensors)[None, :, :]
    raw = float(bundle.predict(window)[0, 0])
    rul = max(0.0, raw)
    score, threshold = None, None
    if anomaly is not None:
        score = float(np.mean((anomaly.predict(window) - window) ** 2))
        threshold = float(anomaly.manifest["anomaly_threshold"])
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid anomaly calibration")
    recommendation, reasons, state = rule_config.apply(
        rul,
        last.cycle,
        score > threshold if score is not None and threshold is not None else None,
        prior,
    )
    result.update(
        status="ready",
        raw_rul_cycles=raw,
        rul_cycles=rul,
        anomaly_score=score,
        anomaly_threshold=threshold,
        health_index=(
            100 * min(rul / bundle.config.rul_cap_cycles, 1)
            if bundle.config.rul_cap_cycles is not None
            else None
        ),
        recommendation=recommendation,
        reason_codes=reasons,
    )
    return Prediction.model_validate(result), state
