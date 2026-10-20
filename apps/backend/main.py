"""Local research API with verified models, separate truth, and generation-aware replay."""

import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from neuropulse.data.acquire import sha256
from neuropulse.data.prepare import verify_raw
from neuropulse.data.readers import SENSOR_NAMES, read_engines
from neuropulse.decisions.rules import DemoRules
from neuropulse.inference.bundles import ModelBundle
from neuropulse.inference.service import Observation, Prediction, infer_prefix
from neuropulse.replay.session import ReplaySession

ROOT = Path(__file__).resolve().parents[2]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PredictionRequest(Strict):
    model_version: str
    anomaly_model_version: str | None = None
    observations: list[Observation] = Field(min_length=1, max_length=100)


class ReplayRequest(Strict):
    machine_id: str
    model_version: str
    anomaly_model_version: str | None = None
    speed_cps: float = Field(default=1.0, ge=0.1, le=10)


class Control(Strict):
    action: Literal["start", "pause", "resume", "reset"]
    generation: int = Field(ge=0)
    speed_cps: float | None = Field(default=None, ge=0.1, le=10)


class Step(Strict):
    step_id: str = Field(min_length=1, max_length=100)
    generation: int = Field(ge=0)


def create_app(root: Path = ROOT) -> FastAPI:
    app = FastAPI(title="NeuroPulse research demonstration", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.getenv("NEUROPULSE_FRONTEND_ORIGIN", "http://localhost:3000")],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    raw = Path(os.getenv("NEUROPULSE_DATA_DIR", str(root / "data"))) / "raw/cmapss"
    artifacts = Path(os.getenv("NEUROPULSE_ARTIFACT_DIR", str(root / "artifacts/models")))
    results = Path(os.getenv("NEUROPULSE_RESULTS_DIR", str(root / "results")))
    rule_file = root / "configs/maintenance.yaml"
    if rule_file.exists():
        import yaml

        values = yaml.safe_load(rule_file.read_text())
        values["version"] += "-" + sha256(rule_file.read_bytes())[:8]
        rules = DemoRules(**values)
    else:
        rules = DemoRules()
    bundles: dict[str, ModelBundle] = {}
    sessions: dict[str, ReplaySession] = {}
    machines = {}
    data_error = None
    try:
        verify_raw(raw)
        all_engines = read_engines(raw / "train_FD001.txt")
        preparation = json.loads((root / "data/processed/FD001/manifest.json").read_text())
        for partition in ("train", "validation"):
            for i in preparation[f"{partition}_ids"]:
                machines[f"{partition}-fd001-{i}"] = all_engines[i]
    except (OSError, ValueError, KeyError) as exc:
        data_error = str(exc)

    def identifier(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise HTTPException(404, "Unknown resource")
        return value

    def bundle(version: str) -> ModelBundle:
        identifier(version)
        if version not in bundles:
            path = artifacts / version
            if not (path / "manifest.json").exists():
                raise HTTPException(404, "Unknown model version")
            try:
                loaded = ModelBundle(path)
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(503, f"Model artifact unavailable: {exc}") from exc
            if len(bundles) >= 4:
                bundles.pop(next(iter(bundles)))
            bundles[version] = loaded
        return bundles[version]

    def machine(machine_id: str):
        if machine_id not in machines:
            raise HTTPException(404, "Unknown machine")
        return machines[machine_id]

    def session(session_id: str) -> ReplaySession:
        active = sessions.get(session_id)
        if active is None:
            raise HTTPException(404, "Unknown session")
        if time.monotonic() - active.touched > 3600:
            sessions.pop(session_id)
            raise HTTPException(404, "Replay session expired")
        return active

    def prefix_end(machine_id: str, session_id: str | None, available: int) -> int:
        if session_id is None:
            return available
        active = session(session_id)
        if active.machine_id != machine_id:
            raise HTTPException(409, "Session/machine mismatch")
        return active.cursor

    def page(items: list, cursor: int, limit: int):
        return {
            "items": items[cursor : cursor + limit],
            "next_cursor": cursor + limit if cursor + limit < len(items) else None,
            "source": "historical_replay",
        }

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request.state.request_id = str(uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logging.getLogger("neuropulse.api").info(
            "%s %s %s %.4f",
            request.state.request_id,
            request.method,
            request.url.path,
            time.perf_counter() - start,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "request_id": getattr(request.state, "request_id", ""),
                    "details": {},
                }
            },
        )

    @app.exception_handler(ValueError)
    async def domain_error(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "CONTRACT_CONFLICT",
                    "message": str(exc),
                    "request_id": getattr(request.state, "request_id", ""),
                    "details": {},
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Request validation failed",
                    "request_id": getattr(request.state, "request_id", ""),
                    "details": {
                        "issues": [
                            {"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
                            for e in exc.errors()
                        ]
                    },
                }
            },
        )

    @app.get("/api/v1/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/api/v1/health/ready")
    def ready():
        if data_error:
            raise HTTPException(503, f"Dataset unavailable: {data_error}")
        selection = results / "deployment.json"
        if not selection.exists() and not os.getenv("NEUROPULSE_RUL_MODEL_VERSION"):
            raise HTTPException(503, "No validation-selected deployment model configured")
        chosen = json.loads(selection.read_text()) if selection.exists() else {}
        version = os.getenv("NEUROPULSE_RUL_MODEL_VERSION") or chosen["model_version"]
        bundle(version)
        anomaly_version = os.getenv("NEUROPULSE_ANOMALY_MODEL_VERSION") or chosen.get(
            "anomaly_model_version"
        )
        if anomaly_version and not bundle(version).compatible(bundle(anomaly_version)):
            raise HTTPException(503, "Configured branches are incompatible")
        return {
            "status": "ready",
            "model_version": version,
            "anomaly_model_version": anomaly_version,
            "dataset": "cmapss",
            "subset": "FD001",
            "source": "historical_replay",
        }

    @app.get("/api/v1/machines")
    def list_machines(
        partition: str = "validation",
        cursor: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        items = [
            {
                "machine_id": key,
                "engine_id": int(rows[0, 0]),
                "partition": key.split("-")[0],
                "source": "historical_replay",
                "available_cycles": len(rows),
            }
            for key, rows in machines.items()
            if key.startswith(partition + "-")
        ]
        return page(items, cursor, limit)

    @app.get("/api/v1/machines/{machine_id}")
    def machine_detail(machine_id: str, session_id: str | None = None):
        rows = machine(machine_id)
        return {
            "machine_id": machine_id,
            "source": "historical_replay",
            "sensors": SENSOR_NAMES,
            "cycle": prefix_end(machine_id, session_id, len(rows)),
        }

    @app.get("/api/v1/machines/{machine_id}/sensors")
    def sensors(
        machine_id: str,
        session_id: str | None = None,
        from_cycle: int = Query(1, ge=1),
        to_cycle: int | None = Query(None, ge=1),
        cursor: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        rows = machine(machine_id)
        end = min(to_cycle or len(rows), prefix_end(machine_id, session_id, len(rows)))
        points = [
            {"cycle": int(row[1]), **dict(zip(SENSOR_NAMES, row[5:].tolist(), strict=True))}
            for row in rows[from_cycle - 1 : end]
        ]
        return page(points, cursor, limit)

    @app.get("/api/v1/machines/{machine_id}/evaluation")
    def evaluation(machine_id: str, session_id: str | None = None):
        rows = machine(machine_id)
        end = prefix_end(machine_id, session_id, len(rows))
        return {
            "items": [
                {"cycle": int(row[1]), "true_rul_cycles": len(rows) - int(row[1])}
                for row in rows[:end]
            ],
            "label_source": "run_to_failure_evaluation_only",
        }

    @app.post("/api/v1/predictions", response_model=Prediction)
    def predict(request: PredictionRequest):
        result, _ = infer_prefix(
            request.observations,
            bundle(request.model_version),
            bundle(request.anomaly_model_version) if request.anomaly_model_version else None,
            rules=rules,
        )
        return result

    @app.post("/api/v1/replays", status_code=201)
    def create_replay(request: ReplayRequest):
        for key in list(sessions):
            if time.monotonic() - sessions[key].touched > 3600:
                sessions.pop(key)
        if len(sessions) >= 10:
            raise HTTPException(409, "Maximum 10 active replay sessions")
        active = ReplaySession(
            machine(request.machine_id),
            request.machine_id,
            bundle(request.model_version),
            bundle(request.anomaly_model_version) if request.anomaly_model_version else None,
            request.speed_cps,
            results / "runtime",
            rules=rules,
        )
        sessions[active.id] = active
        return active.snapshot()

    @app.get("/api/v1/replays/{session_id}")
    def get_replay(session_id: str):
        return session(session_id).snapshot()

    @app.post("/api/v1/replays/{session_id}/control")
    def control(session_id: str, request: Control):
        return session(session_id).control(request.action, request.generation, request.speed_cps)

    @app.post("/api/v1/replays/{session_id}/step")
    def step(session_id: str, request: Step):
        return session(session_id).step(request.step_id, request.generation)

    @app.delete("/api/v1/replays/{session_id}", status_code=204)
    def delete_replay(session_id: str):
        session(session_id)
        sessions.pop(session_id)

    @app.get("/api/v1/machines/{machine_id}/predictions")
    def history(
        machine_id: str,
        session_id: str | None = None,
        cursor: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        machine(machine_id)
        active = session(session_id) if session_id else None
        if active and active.machine_id != machine_id:
            raise HTTPException(409, "Session/machine mismatch")
        items = (
            active.predictions
            if active
            else [p for s in sessions.values() if s.machine_id == machine_id for p in s.predictions]
        )
        return page(items, cursor, limit)

    @app.get("/api/v1/alerts")
    def alerts(
        session_id: str | None = None,
        cursor: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        items = (
            session(session_id).alerts
            if session_id
            else [a for s in sessions.values() for a in s.alerts]
        )
        return page(items, cursor, limit)

    @app.get("/api/v1/models")
    def models():
        return {
            "items": [json.loads(p.read_text()) for p in sorted(artifacts.glob("*/manifest.json"))]
        }

    @app.get("/api/v1/models/{version}")
    def model(version: str):
        return bundle(version).manifest

    @app.get("/api/v1/experiments")
    def experiments():
        items = []
        for path in sorted(results.glob("*/manifest.json")):
            document = json.loads(path.read_text())
            if (
                isinstance(document, dict)
                and document.get("run_id") == path.parent.name
                and isinstance(document.get("config"), dict)
                and isinstance(document.get("metrics"), dict)
            ):
                items.append(document)
        return {"items": items}

    @app.get("/api/v1/experiments/{run_id}")
    def experiment(run_id: str):
        directory = results / identifier(run_id)
        path = directory / "manifest.json"
        if not path.exists():
            raise HTTPException(404, "Unknown experiment")
        document = json.loads(path.read_text())
        if (
            not isinstance(document, dict)
            or document.get("run_id") != run_id
            or not isinstance(document.get("config"), dict)
            or not isinstance(document.get("metrics"), dict)
        ):
            raise HTTPException(404, "Unknown experiment")
        for name in ("history.csv", "predictions.csv"):
            path = directory / name
            if path.exists():
                with path.open() as stream:
                    document[name] = list(csv.DictReader(stream))
        document["artifacts"] = [
            p.name for p in directory.iterdir() if p.suffix in (".csv", ".json", ".yaml")
        ]
        return document

    @app.get("/api/v1/experiments/{run_id}/artifacts/{artifact_id}")
    def artifact(run_id: str, artifact_id: str):
        allowed = {
            "metrics.json",
            "history.csv",
            "predictions.csv",
            "config.resolved.yaml",
            "environment.json",
            "timing.json",
            "manifest.json",
        }
        path = results / identifier(run_id) / artifact_id
        if artifact_id not in allowed or not path.is_file():
            raise HTTPException(404, "Unknown artifact")
        return FileResponse(path)

    app.state.sessions, app.state.bundles = sessions, bundles
    return app


app = create_app()
