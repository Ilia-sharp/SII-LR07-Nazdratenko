"""Общий объект прогресса длительных операций (обучение, оценка).

Фоновые задачи пишут сюда состояние, а эндпоинт /api/progress отдаёт его сайту.
Одновременно может выполняться только одна задача.
"""

import copy
import threading
import time

_lock = threading.Lock()


def _empty() -> dict:
    return {
        "job_type": None,
        "stage": "idle",
        "done": 0,
        "total": 0,
        "percent": 0.0,
        "mean_reward": None,
        "eta_seconds": None,
        "message": "Ничего не выполняется",
        "started_at": None,
        "finished_at": None,
        "error": None,
    }


_state = _empty()
_started_monotonic = None


class JobBusyError(RuntimeError):
    """Уже выполняется другая операция."""


def start_job(job_type: str, total: int, stage: str = "init") -> None:
    """Начинает задачу. Если другая задача ещё идёт — бросает JobBusyError."""
    global _started_monotonic
    with _lock:
        if _state["stage"] not in ("idle", "done", "error"):
            raise JobBusyError("Уже выполняется другая операция")
        _state.update(_empty())
        _state.update(
            job_type=job_type,
            stage=stage,
            total=int(total),
            message="Запуск…",
            started_at=time.time(),
        )
        _started_monotonic = time.monotonic()


def update(done: int | None = None, stage: str | None = None,
           mean_reward: float | None = None, message: str | None = None) -> None:
    """Обновляет процент, ETA и текущий этап."""
    with _lock:
        if stage is not None:
            _state["stage"] = stage
        if done is not None:
            _state["done"] = int(done)
            total = _state["total"] or 1
            _state["percent"] = round(min(100.0, 100.0 * done / total), 2)
            if _started_monotonic and done > 0:
                elapsed = time.monotonic() - _started_monotonic
                remaining = elapsed / done * max(0, total - done)
                _state["eta_seconds"] = round(remaining, 1)
        if mean_reward is not None:
            _state["mean_reward"] = round(float(mean_reward), 2)
        if message is not None:
            _state["message"] = message


def finish(result: dict | None = None) -> None:
    """Завершает задачу успешно."""
    with _lock:
        _state.update(stage="done", percent=100.0, eta_seconds=0.0,
                      message="Готово", finished_at=time.time(), error=None)
        if result is not None:
            _state["result"] = result


def fail(message: str) -> None:
    """Завершает задачу с ошибкой (сообщение на русском, без трейсбеков)."""
    with _lock:
        _state.update(stage="error", error=message, message=message,
                      finished_at=time.time())


def snapshot() -> dict:
    """Копия текущего состояния для ответа API."""
    with _lock:
        return copy.deepcopy(_state)
