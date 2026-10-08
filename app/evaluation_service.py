"""Оценка агента: PPO (greedy) против случайного агента на одинаковых seed эпизодов.

Используется и scripts/evaluate.py, и кнопкой на сайте (POST /api/evaluate).
Модуль не импортирует torch и stable_baselines3 на верхнем уровне:
  - для API используется numpy-политика (app/policy.py);
  - для scripts/evaluate.py SB3 подгружается внутри функции.
"""

import json
import os
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from config import (
    ACTION_MODE,
    ENV_ID,
    EVAL_EPISODES,
    EVALUATION_JSON_PATH,
    MODEL_VERSION,
    SEED,
)

ProgressCallback = Callable[[int, int], None]


def _run_episode(env, policy_fn, episode_seed: int) -> int:
    """Один эпизод. Награда CartPole = число шагов (максимум 500)."""
    obs, _ = env.reset(seed=episode_seed)
    total = 0
    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += int(reward)
        if terminated or truncated:
            break
    return total


def _stats(rewards: list[int]) -> dict:
    arr = np.asarray(rewards, dtype=float)
    return {
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "rewards": [int(r) for r in rewards],
    }


def evaluate_agents(ppo_policy_fn, progress_callback: ProgressCallback | None = None) -> dict:
    """Запускает PPO и случайного агента на одинаковых seed и возвращает итоговый словарь.

    ppo_policy_fn(obs) -> int. Для scripts/evaluate.py это model.predict (deterministic=True),
    для API — numpy-политика. Оба варианта должны давать одинаковые числа.
    """
    import gymnasium as gym  # ленивый импорт: старт сервиса не требует gymnasium

    env = gym.make(ENV_ID)
    total_steps = EVAL_EPISODES * 2
    done_steps = 0

    ppo_rewards = []
    for i in range(EVAL_EPISODES):
        ppo_rewards.append(_run_episode(env, ppo_policy_fn, SEED + i))
        done_steps += 1
        if progress_callback:
            progress_callback(done_steps, total_steps)

    env.action_space.seed(SEED)
    random_rewards = []
    for i in range(EVAL_EPISODES):
        random_rewards.append(_run_episode(env, lambda _obs: env.action_space.sample(), SEED + i))
        done_steps += 1
        if progress_callback:
            progress_callback(done_steps, total_steps)
    env.close()

    ppo = _stats(ppo_rewards)
    rnd = _stats(random_rewards)
    ratio = round(ppo["mean"] / rnd["mean"], 2) if rnd["mean"] > 0 else None
    return {
        "seed": SEED,
        "episodes": EVAL_EPISODES,
        "action_mode": ACTION_MODE,
        "model_version": MODEL_VERSION,
        "ppo": ppo,
        "random": rnd,
        "improvement_ratio": ratio,
        "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def save_result(result: dict, path: str = EVALUATION_JSON_PATH) -> None:
    """Сохраняет results/evaluation.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def load_result(path: str = EVALUATION_JSON_PATH) -> dict | None:
    """Читает results/evaluation.json, если он есть."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def print_table(result: dict) -> None:
    """Таблица в консоль: агент | среднее | std | min | max | эпизодов."""
    print(f"{'агент':<10}| {'среднее':>8} | {'std':>6} | {'min':>5} | {'max':>5} | эпизодов")
    print("-" * 60)
    for name, key in (("PPO", "ppo"), ("Random", "random")):
        s = result[key]
        print(f"{name:<10}| {s['mean']:>8.2f} | {s['std']:>6.2f} | {s['min']:>5} | {s['max']:>5} | {result['episodes']}")
    ratio = result["improvement_ratio"]
    if ratio is not None:
        print(f"\nPPO лучше случайного в {ratio} раз.")
