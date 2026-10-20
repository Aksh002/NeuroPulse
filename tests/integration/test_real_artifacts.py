"""Optional local gate: a frozen real bundle agrees with chronological API replay."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.backend.main import create_app
from neuropulse.inference.bundles import ModelBundle


@pytest.mark.dataset
def test_real_bundle_matches_api_at_first_complete_prefix():
    root = Path(__file__).resolve().parents[2]
    deployment = root / "results/deployment.json"
    if not deployment.exists() or not (root / "data/processed/FD001/manifest.json").exists():
        pytest.skip("Local prepared FD001 data and selected bundle are unavailable")
    versions = json.loads(deployment.read_text())
    model = ModelBundle(root / "artifacts/models" / versions["model_version"])
    with TestClient(create_app(root)) as client:
        ready = client.get("/api/v1/health/ready")
        assert ready.status_code == 200
        machine = client.get("/api/v1/machines").json()["items"][0]["machine_id"]
        response = client.post(
            "/api/v1/replays",
            json={
                "machine_id": machine,
                "model_version": versions["model_version"],
                "anomaly_model_version": versions["anomaly_model_version"],
            },
        )
        assert response.status_code == 201
        session = response.json()["session_id"]
        endpoint = f"/api/v1/replays/{session}"
        assert (
            client.post(
                endpoint + "/control", json={"action": "start", "generation": 0}
            ).status_code
            == 200
        )
        for cycle in range(1, model.config.window_length + 1):
            step = client.post(endpoint + "/step", json={"step_id": str(cycle), "generation": 0})
            assert step.status_code == 200
            prediction = step.json()["last_prediction"]
            if cycle < model.config.window_length:
                assert prediction["status"] == "warming_up"
                assert prediction["rul_cycles"] is None
        assert prediction["status"] == "ready"
        assert prediction["anomaly_score"] is not None
        sensors = client.get(
            f"/api/v1/machines/{machine}/sensors?session_id={session}&limit=100"
        ).json()["items"]
        assert len(sensors) == model.config.window_length
        raw = np.array(
            [[row[f"sensor_{i:02d}"] for i in range(1, 22)] for row in sensors],
            dtype=np.float64,
        )
        expected = float(
            np.maximum(0, model.predict(model.scaler.transform(raw)[None, :, :])[0, 0])
        )
        assert prediction["rul_cycles"] == pytest.approx(expected, abs=1e-8)
        assert "true_rul_cycles" not in step.json()["observation"]
        assert client.delete(endpoint).status_code == 204
