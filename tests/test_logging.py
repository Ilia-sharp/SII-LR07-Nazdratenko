"""Журнал logs/actions.jsonl: после POST /act появляется строка с нужными полями."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from config import ACTIONS_LOG_PATH, NUMPY_POLICY_PATH

client = TestClient(app)


@pytest.mark.skipif(not os.path.exists(NUMPY_POLICY_PATH), reason="Нет models/policy_v1.0.npz")
def test_act_writes_log_line(tmp_path, monkeypatch):
    from app import action_logger

    log_file = tmp_path / "actions.jsonl"
    monkeypatch.setattr(action_logger, "ACTIONS_LOG_PATH", str(log_file))

    before = log_file.read_text(encoding="utf-8").splitlines() if log_file.exists() else []
    r = client.post("/act", json={"state": [0.01, 0.2, -0.03, -0.4], "reward": 1.0})
    assert r.status_code == 200

    record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert {"time", "state", "action", "reward", "model_version"} <= set(record.keys())
    assert record["time"].endswith("Z")
    assert record["state"] == [0.01, 0.2, -0.03, -0.4]
    assert record["reward"] == 1.0
    assert record["action"] in (0, 1)
    assert len(before) + 1 == len(log_file.read_text(encoding="utf-8").splitlines())


def test_record_format_without_reward():
    from app.action_logger import build_record

    rec = build_record([0.1, 0.2, 0.3, 0.4], 1, None)
    assert rec["reward"] is None
    assert rec["time"].endswith("Z")
    assert rec["model_version"] == "ppo-cartpole-v1.0"
