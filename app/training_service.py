"""Фоновое обучение-кандидат для сайта.

Импорт torch и stable_baselines3 находится ВНУТРИ функций, поэтому старт сервера
и /act их не загружают. Итоговая модель не подменяет боевую: кандидат пишется
в models/candidate/ и сравнивается с текущей политикой.
"""

import os
import random

import numpy as np

from app import progress
from config import (
    CANDIDATE_DIR,
    ENV_ID,
    EXPECTED_MIN_MEAN_REWARD,
    SEED,
)


def run_training_candidate(timesteps: int) -> dict:
    """Обучает PPO на timesteps шагов, сохраняет кандидат и возвращает сводку."""
    # Тяжёлые импорты только здесь:
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from app.policy_export import export_numpy_policy
    from app.evaluation_service import _run_episode  # noqa: PLC2701 — общий код эпизода

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    progress.update(stage="train", message="Обучение PPO…")

    class ProgressCallback(BaseCallback):
        def __init__(self, total: int):
            super().__init__()
            self.total = total

        def _on_step(self) -> bool:
            if self.num_timesteps % 1000 == 0 or self.num_timesteps >= self.total:
                ep = self.model.ep_info_buffer
                mean_r = (sum(e["r"] for e in ep) / len(ep)) if len(ep) > 0 else None
                progress.update(done=min(self.num_timesteps, self.total), stage="train",
                                mean_reward=mean_r, message="Обучение PPO…")
            return True

    env = gym.make(ENV_ID)
    env.reset(seed=SEED)
    model = PPO("MlpPolicy", env, seed=SEED, device="cpu", verbose=0)
    model.learn(total_timesteps=timesteps, callback=ProgressCallback(timesteps))

    progress.update(stage="evaluate", done=timesteps, message="Оценка кандидата…")
    eval_rewards = []
    for i in range(20):
        eval_rewards.append(_run_episode(env, lambda obs: int(model.predict(obs, deterministic=True)[0]), SEED + i))
    env.close()
    mean_reward = float(np.mean(eval_rewards))

    zip_path = os.path.join(CANDIDATE_DIR, "ppo_candidate.zip")
    npz_path = os.path.join(CANDIDATE_DIR, "policy_candidate.npz")
    model.save(zip_path)
    export_numpy_policy(model, npz_path)

    return {
        "timesteps": timesteps,
        "candidate_mean_reward": round(mean_reward, 2),
        "meets_expected_quality": mean_reward >= EXPECTED_MIN_MEAN_REWARD,
        "candidate_path": zip_path,
    }
