"""Журнал logs/actions.jsonl: после POST /act появляется строка с нужными полями."""

import json
import os

import pytest
from fastapi.testclient import TestClient

import config
from app.main import app
from config import NUMPY_POLICY_PATH

client = TestClient(app)


@pytest.mark.skipif(not os.path.exists(NUMPY_POLICY_PATH), reason="Нет models/policy_v1.0.npz")
def test_act_writes_log_line():
    log_file = config.ACTIONS_LOG_PATH  # временный файл из conftest
    assert not os.path.exists(log_file)

    r = client.post("/act", json={"state": [0.01, 0.2, -0.03, -0.4], "reward": 1.0})
    assert r.status_code == 200

    with open(log_file, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert {"time", "state", "action", "reward", "model_version"} <= set(record.keys())
    assert record["time"].endswith("Z")
    assert record["state"] == [0.01, 0.2, -0.03, -0.4]
    assert record["reward"] == 1.0
    assert record["action"] in (0, 1)
    assert record["model_version"] == "ppo-cartpole-v1.0"


@pytest.mark.skipif(not os.path.exists(NUMPY_POLICY_PATH), reason="Нет models/policy_v1.0.npz")
def test_reward_is_null_when_not_sent():
    client.post("/act", json={"state": [0.0, 0.0, 0.0, 0.0]})
    with open(config.ACTIONS_LOG_PATH, encoding="utf-8") as fh:
        record = json.loads(fh.read().splitlines()[-1])
    assert record["reward"] is None


def test_record_format_without_reward():
    from app.action_logger import build_record

    rec = build_record([0.1, 0.2, 0.3, 0.4], 1, None)
    assert rec["reward"] is None
    assert rec["time"].endswith("Z")
    assert rec["model_version"] == "ppo-cartpole-v1.0"


def test_logs_endpoint_reads_written_records():
    from app import action_logger

    action_logger.append(action_logger.build_record([0, 0, 0, 0], 1, 1.0))
    items = client.get("/api/logs?limit=5").json()["items"]
    assert len(items) == 1 and items[0]["action"] == 1
