"""Numpy-инференс политики. Не импортирует torch и stable_baselines3.

Веса извлекаются скриптом scripts/train.py в файл models/policy_v1.0.npz.
Архитектура: 4 -> 64 (Tanh) -> 64 (Tanh) -> 2 логита. Действие = argmax логитов.
"""

import os
import threading

import numpy as np

from config import NUMPY_POLICY_PATH

_lock = threading.Lock()
_weights = None


class PolicyNotLoadedError(RuntimeError):
    """Файл весов не найден или не читается."""


def load_weights(path: str = NUMPY_POLICY_PATH) -> dict:
    """Загружает веса из npz один раз и кэширует их."""
    global _weights
    with _lock:
        if _weights is not None:
            return _weights
        if not os.path.exists(path):
            raise PolicyNotLoadedError(
                "Файл весов модели не найден. Выполните: python scripts/train.py"
            )
        try:
            data = np.load(path)
            _weights = {key: data[key].astype(np.float64) for key in data.files}
        except Exception as exc:  # noqa: BLE001 — любая ошибка чтения → понятное сообщение
            raise PolicyNotLoadedError(f"Не удалось прочитать веса модели: {exc}") from exc
        return _weights


def is_loaded() -> bool:
    """Загружены ли веса (без попытки загрузки)."""
    return _weights is not None


def reset_cache() -> None:
    """Сбрасывает кэш весов (используется в тестах)."""
    global _weights
    with _lock:
        _weights = None


def logits(state, weights: dict | None = None) -> np.ndarray:
    """Логиты для одного состояния (массив из 4 чисел)."""
    w = weights if weights is not None else load_weights()
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    h1 = np.tanh(x @ w["W1"].T + w["b1"])
    h2 = np.tanh(h1 @ w["W2"].T + w["b2"])
    return h2 @ w["W3"].T + w["b3"]


def act(state, weights: dict | None = None) -> int:
    """Greedy-действие: индекс максимального логита (0 — влево, 1 — вправо)."""
    return int(np.argmax(logits(state, weights)))
