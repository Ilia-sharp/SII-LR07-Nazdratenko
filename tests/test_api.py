"""Тесты контракта API: /health и POST /act."""

import pytest
from fastapi.testclient import TestClient

from app import policy
from app.main import app
from config import MODEL_VERSION, NUMPY_POLICY_PATH

client = TestClient(app)


def _weights_exist() -> bool:
    import os
    return os.path.exists(NUMPY_POLICY_PATH)


def test_health_returns_fields():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "ppo-cartpole-v1.0"
    assert "model_loaded" in data


@pytest.mark.skipif(not _weights_exist(), reason="Нет models/policy_v1.0.npz (запустите scripts/train.py)")
def test_act_contract():
    r = client.post("/act", json={"state": [0.01, 0.2, -0.03, -0.4]})
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"action", "model_version"}
    assert data["action"] in (0, 1)
    assert data["model_version"] == MODEL_VERSION


@pytest.mark.skipif(not _weights_exist(), reason="Нет models/policy_v1.0.npz (запустите scripts/train.py)")
def test_act_accepts_optional_reward():
    r = client.post("/act", json={"state": [0.0, 0.0, 0.0, 0.0], "reward": 1.0})
    assert r.status_code == 200


def test_act_wrong_length_422():
    r = client.post("/act", json={"state": [0.1, 0.2, 0.3]})
    assert r.status_code == 422
    assert "4" in r.json()["detail"]


def test_act_nan_422():
    r = client.post("/act", content='{"state": [NaN, 0, 0, 0]}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_act_inf_422():
    r = client.post("/act", content='{"state": [Infinity, 0, 0, 0]}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_act_missing_state_422():
    r = client.post("/act", json={})
    assert r.status_code == 422


def test_act_without_weights_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "_weights", None)
    monkeypatch.setattr(policy, "NUMPY_POLICY_PATH", str(tmp_path / "missing.npz"))

    def fake_load(path=str(tmp_path / "missing.npz")):
        raise policy.PolicyNotLoadedError("Файл весов модели не найден. Выполните: python scripts/train.py")

    monkeypatch.setattr(policy, "load_weights", fake_load)
    r = client.post("/act", json={"state": [0, 0, 0, 0]})
    assert r.status_code == 503
    assert "весов" in r.json()["detail"]
