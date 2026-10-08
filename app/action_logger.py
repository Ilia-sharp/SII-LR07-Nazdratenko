"""Журнал взаимодействий logs/actions.jsonl (потокобезопасная запись в режиме append)."""

import json
import os
import sys
import threading
from datetime import datetime, timezone

from config import ACTIONS_LOG_PATH, MODEL_VERSION

_lock = threading.Lock()


def utc_now_iso() -> str:
    """Время в UTC, ISO 8601, с суффиксом Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_record(state, action: int, reward, model_version: str = MODEL_VERSION) -> dict:
    """Формирует запись журнала с полями time, state, action, reward, model_version."""
    return {
        "time": utc_now_iso(),
        "state": [float(x) for x in state],
        "action": int(action),
        "reward": None if reward is None else float(reward),
        "model_version": model_version,
    }


def append(record: dict, path: str = ACTIONS_LOG_PATH) -> bool:
    """Дописывает одну строку JSON. Ошибку не пробрасывает: при сбое пишет в stderr."""
    try:
        with _lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as exc:
        print(f"Не удалось записать журнал действий: {exc}", file=sys.stderr)
        return False


def read_last(limit: int = 10, path: str = ACTIONS_LOG_PATH) -> list[dict]:
    """Последние записи журнала (от новых к старым). Битые строки пропускаются."""
    limit = max(1, min(int(limit), 100))
    if not os.path.exists(path):
        return []
    with _lock:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    records = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(records) >= limit:
            break
    return records
