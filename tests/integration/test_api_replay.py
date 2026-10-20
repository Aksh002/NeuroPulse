"""Original synthetic fixtures: HTTP, truth isolation, replay retries and prefix invariance."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.backend.main import create_app
from neuropulse.config import TrainConfig
from neuropulse.data.acquire import sha256
from neuropulse.data.preprocessing import SensorScaler
from neuropulse.data.readers import SENSOR_NAMES
from neuropulse.decisions.rules import DemoRules
from neuropulse.inference.bundles import ModelBundle
from neuropulse.inference.service import Observation, infer_prefix
from neuropulse.models.numpy_mlp import build_mlp
from neuropulse.nn_numpy.serialization import save
from neuropulse.replay.session import ReplaySession


@pytest.fixture
def fixture_root(tmp_path, monkeypatch):
    for variable in (
        "NEUROPULSE_DATA_DIR",
        "NEUROPULSE_ARTIFACT_DIR",
        "NEUROPULSE_RESULTS_DIR",
        "NEUROPULSE_RUL_MODEL_VERSION",
        "NEUROPULSE_ANOMALY_MODEL_VERSION",
    ):
        monkeypatch.delenv(variable, raising=False)
    raw = tmp_path / "data/raw/cmapss"
    raw.mkdir(parents=True)
    engines = {}
    for i in (1, 2):
        rows = np.zeros((40, 26))
        rows[:, 0], rows[:, 1] = i, np.arange(1, 41)
        rows[:, 5:] = np.arange(40)[:, None] + np.arange(21)[None, :]
        engines[i] = rows
    for kind in ("train", "test"):
        np.savetxt(raw / f"{kind}_FD001.txt", np.concatenate(list(engines.values())))
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    f"{kind}_FD001.txt": {
                        "sha256": sha256((raw / f"{kind}_FD001.txt").read_bytes())
                    }
                    for kind in ("train", "test")
                }
            }
        )
    )
    processed = tmp_path / "data/processed/FD001"
    processed.mkdir(parents=True)
    (processed / "manifest.json").write_text(json.dumps({"train_ids": [1], "validation_ids": [2]}))
    config = TrainConfig(family="numpy_mlp", window_length=3, dense_width=8, hidden_layers=1)
    model = build_mlp(21, config)
    for name, parameter in model.named_parameters().items():
        parameter.data.fill(0.001 if "weight" in name else 1)
    model.layers[-1].bias.data.fill(48)
    directory = tmp_path / "artifacts/models/synthetic-model"
    directory.mkdir(parents=True)
    save(model, directory / "weights")
    SensorScaler.fit(engines, [1]).save(directory / "scaler.npz")
    meta = {
        "schema_version": "1.0",
        "dataset": "cmapss",
        "subset": "FD001",
        "model_version": "synthetic-model",
        "backend": "numpy",
        "family": "numpy_mlp",
        "config": config.model_dump(),
        "weights": "weights.npz",
        "training_ids": [1],
        "features": list(SENSOR_NAMES),
        "checksums": {
            name: sha256((directory / name).read_bytes())
            for name in ("weights.npz", "weights.json", "scaler.npz")
        },
    }
    (directory / "manifest.json").write_text(json.dumps(meta))
    results = tmp_path / "results"
    results.mkdir()
    (results / "deployment.json").write_text(json.dumps({"model_version": "synthetic-model"}))
    return tmp_path, engines


def test_uncapped_model_has_no_implied_health_reference(fixture_root):
    root, engines = fixture_root
    directory = root / "artifacts/models/synthetic-model"
    path = directory / "manifest.json"
    document = json.loads(path.read_text())
    document["config"]["rul_cap_cycles"] = None
    path.write_text(json.dumps(document))
    bundle = ModelBundle(directory)
    observations = [
        Observation(
            engine_id=2,
            cycle=int(row[1]),
            operating_settings=row[2:5].tolist(),
            sensors=row[5:].tolist(),
        )
        for row in engines[2][:3]
    ]
    prediction, _ = infer_prefix(observations, bundle)
    assert prediction.status == "ready"
    assert prediction.health_index is None


def test_http_replay_generation_pause_retry_truth_and_logging(fixture_root):
    root, _ = fixture_root
    with TestClient(create_app(root)) as client:
        assert client.get("/api/v1/health/ready").status_code == 200
        created = client.post(
            "/api/v1/replays",
            json={"machine_id": "validation-fd001-2", "model_version": "synthetic-model"},
        ).json()
        identifier = created["session_id"]
        path = f"/api/v1/replays/{identifier}"
        assert (
            client.post(path + "/step", json={"step_id": "1", "generation": 0}).status_code == 409
        )
        client.post(path + "/control", json={"action": "start", "generation": 0})
        for index in range(1, 4):
            body = {"step_id": str(index), "generation": 0}
            response = client.post(path + "/step", json=body).json()
            assert client.post(path + "/step", json=body).json() == response
        assert response["prediction"]["status"] == "ready"
        assert "true_rul_cycles" not in response["prediction"]
        assert "true_rul_cycles" not in response["observation"]
        query = f"?session_id={identifier}"
        assert (
            len(client.get("/api/v1/machines/validation-fd001-2/sensors" + query).json()["items"])
            == 3
        )
        assert (
            len(
                client.get("/api/v1/machines/validation-fd001-2/evaluation" + query).json()["items"]
            )
            == 3
        )
        client.post(path + "/control", json={"action": "pause", "generation": 0})
        assert (
            client.post(path + "/step", json={"step_id": "4", "generation": 0}).status_code == 409
        )
        reset = client.post(path + "/control", json={"action": "reset", "generation": 0}).json()
        assert reset["generation"] == 1 and reset["cycle"] == 0
        assert (
            client.post(path + "/step", json={"step_id": "3", "generation": 0}).status_code == 409
        )
        assert len((root / f"results/runtime/{identifier}.jsonl").read_text().splitlines()) == 3


def test_future_mutation_and_concurrent_retry_cannot_change_current_prediction(fixture_root):
    root, engines = fixture_root
    bundle = ModelBundle(root / "artifacts/models/synthetic-model")
    original, changed = engines[2].copy(), engines[2].copy()
    changed[3:, 5:] = 1e9
    sessions = [ReplaySession(rows, "validation-fd001-2", bundle) for rows in (original, changed)]
    for session in sessions:
        session.control("start", 0)
        session.step("1", 0)
        session.step("2", 0)
    first = sessions[0].step("3", 0)["prediction"]
    with ThreadPoolExecutor(2) as pool:
        replies = list(pool.map(lambda _: sessions[1].step("3", 0), range(2)))
    assert replies[0] == replies[1]
    assert first == replies[0]["prediction"]
    assert sessions[1].cursor == 3


def test_unknown_models_invalid_inputs_and_corrupted_bundle(fixture_root):
    root, _ = fixture_root
    with TestClient(create_app(root)) as client:
        assert client.get("/api/v1/models/unknown").status_code == 404
        response = client.post(
            "/api/v1/predictions",
            json={
                "model_version": "synthetic-model",
                "observations": [
                    {
                        "engine_id": 2,
                        "cycle": 1,
                        "operating_settings": [0.0, 0.0, 0.0],
                        "sensors": [0.0] * 21,
                        "true_rul_cycles": 100,
                    }
                ],
            },
        )
        assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_REQUEST"
    path = root / "artifacts/models/synthetic-model/weights.npz"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ModelBundle(path.parent)


def test_experiment_listing_excludes_other_manifest_formats(fixture_root):
    root, _ = fixture_root
    results = root / "results"
    experiment = results / "valid-run"
    experiment.mkdir()
    (experiment / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "valid-run",
                "family": "mlp",
                "config": {"seed": 42, "window_length": 30},
                "metrics": {"validation": {"rmse_cycles": 12.0}},
            }
        )
    )
    manual = results / "manual-ae"
    manual.mkdir()
    (manual / "manifest.json").write_text(
        json.dumps({"family": "numpy_dense_autoencoder", "seed": 42})
    )
    with TestClient(create_app(root)) as client:
        assert [item["run_id"] for item in client.get("/api/v1/experiments").json()["items"]] == [
            "valid-run"
        ]
        assert client.get("/api/v1/experiments/valid-run").status_code == 200
        assert client.get("/api/v1/experiments/manual-ae").status_code == 404


def test_rules_boundaries_persistence_clearing_and_gap():
    rules = DemoRules()
    assert rules.apply(20, 1, None)[0] == "critical_warning"
    assert rules.apply(50, 1, None)[0] == "maintenance_recommended"
    state = None
    for cycle in range(1, 4):
        status, _, state = rules.apply(70, cycle, True, state)
    assert status == "maintenance_recommended"
    for cycle in range(4, 7):
        status, _, state = rules.apply(90, cycle, False, state)
    assert status == "normal"
    assert rules.apply(70, 99, True, state)[0] == "normal"
