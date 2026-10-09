"""Общие настройки тестов: корень проекта в sys.path и изолированный журнал."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def isolated_actions_log(tmp_path, monkeypatch):
    """Тесты пишут журнал во временный файл, а не в настоящий logs/actions.jsonl."""
    import config

    monkeypatch.setattr(config, "ACTIONS_LOG_PATH", str(tmp_path / "actions.jsonl"))
