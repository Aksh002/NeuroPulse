"""Locked, generation-aware historical replay with idempotent single-cycle stepping."""

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from neuropulse.data.readers import Array
from neuropulse.decisions.rules import DemoRules, RuleState
from neuropulse.inference.bundles import ModelBundle
from neuropulse.inference.service import Observation, infer_prefix


class ReplaySession:
    def __init__(
        self,
        rows: Array,
        machine_id: str,
        bundle: ModelBundle,
        anomaly: ModelBundle | None = None,
        speed: float = 1.0,
        log_dir: Path | None = None,
        rules: DemoRules | None = None,
    ) -> None:
        if not 0.1 <= speed <= 10 or (anomaly is not None and not bundle.compatible(anomaly)):
            raise ValueError("Invalid replay rate/branch compatibility")
        self.id, self.machine_id, self.rows = str(uuid4()), machine_id, rows
        self.bundle, self.anomaly, self.speed, self.log_dir = bundle, anomaly, speed, log_dir
        self.rules = rules or DemoRules()
        self.lock = threading.RLock()
        self.generation, self.cursor, self.status = 0, 0, "paused"
        self.state = RuleState()
        self.predictions: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.receipts: dict[str, dict[str, Any]] = {}
        self.touched = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "machine_id": self.machine_id,
            "generation": self.generation,
            "cycle": self.cursor,
            "status": self.status,
            "speed_cps": self.speed,
            "model_version": self.bundle.manifest["model_version"],
            "anomaly_model_version": self.anomaly.manifest["model_version"]
            if self.anomaly
            else None,
            "last_prediction": self.predictions[-1] if self.predictions else None,
            "source": "historical_replay",
        }

    def control(self, action: str, generation: int, speed: float | None = None) -> dict[str, Any]:
        with self.lock:
            if generation != self.generation:
                raise ValueError("STALE_REPLAY_GENERATION")
            if speed is not None:
                if not 0.1 <= speed <= 10:
                    raise ValueError("Invalid replay rate")
                self.speed = speed
            if action == "reset":
                self.generation += 1
                self.cursor, self.status, self.state = 0, "paused", RuleState()
                self.predictions, self.alerts, self.receipts = [], [], {}
            elif action == "pause":
                if self.status != "completed":
                    self.status = "paused"
            elif action in ("start", "resume"):
                if self.cursor >= len(self.rows):
                    raise ValueError("REPLAY_COMPLETED")
                self.status = "running"
            else:
                raise ValueError("Unknown replay action")
            self.touched = time.monotonic()
            return self.snapshot()

    def step(self, step_id: str, generation: int) -> dict[str, Any]:
        with self.lock:
            if generation != self.generation:
                raise ValueError("STALE_REPLAY_GENERATION")
            if step_id in self.receipts:
                return copy.deepcopy(self.receipts[step_id])
            if self.status != "running":
                raise ValueError("REPLAY_NOT_RUNNING")
            if len(self.receipts) >= 1000:
                raise ValueError("SESSION_CAPACITY_REACHED")
            end = self.cursor + 1
            partition = self.machine_id.split("-")[0]
            observations = [
                Observation.model_validate(
                    {
                        "engine_id": int(row[0]),
                        "cycle": int(row[1]),
                        "partition": partition,
                        "operating_settings": row[2:5].tolist(),
                        "sensors": row[5:].tolist(),
                        "source": "historical_replay",
                    }
                )
                for row in self.rows[max(0, end - self.bundle.config.window_length) : end]
            ]
            prediction, state = infer_prefix(
                observations, self.bundle, self.anomaly, self.state, rules=self.rules
            )
            record = prediction.model_dump()
            alert = None
            if (
                state.recommendation != self.state.recommendation
                and state.recommendation is not None
                and (
                    state.recommendation != "normal"
                    or self.state.recommendation not in (None, "normal")
                )
            ):
                alert = {
                    "session_id": self.id,
                    "generation": self.generation,
                    "cycle": end,
                    "recommendation": state.recommendation,
                    "reason_codes": record["reason_codes"],
                    "rule_version": record["rule_version"],
                    "model_version": record["model_version"],
                }
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / f"{self.id}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "generation": self.generation,
                                "cycle": end,
                                "prediction": record,
                                "alert": alert,
                            }
                        )
                        + "\n"
                    )
            self.cursor, self.state = end, state
            self.predictions.append(record)
            if alert is not None:
                self.alerts.append(alert)
            if end == len(self.rows):
                self.status = "completed"
            self.touched = time.monotonic()
            response = {
                **self.snapshot(),
                "step_id": step_id,
                "prediction": record,
                "observation": observations[-1].model_dump(),
            }
            self.receipts[step_id] = copy.deepcopy(response)
            return response
