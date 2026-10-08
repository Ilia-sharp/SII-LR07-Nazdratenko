"""numpy-политика и SB3 должны давать одинаковый argmax на 1000 состояниях.

Тест пропускается, если torch/stable-baselines3 не установлены или нет файла модели.
"""

import os

import numpy as np
import pytest

from config import SB3_MODEL_PATH, SEED

sb3 = pytest.importorskip("stable_baselines3", reason="stable-baselines3 не установлен")


@pytest.mark.skipif(not os.path.exists(SB3_MODEL_PATH), reason="Нет models/ppo_cartpole_v1.0.zip")
def test_numpy_policy_matches_sb3():
    from app.policy import act, load_weights, reset_cache

    reset_cache()
    weights = load_weights()
    model = sb3.PPO.load(SB3_MODEL_PATH, device="cpu")

    rng = np.random.default_rng(SEED)
    states = rng.uniform(low=-0.05, high=0.05, size=(1000, 4)).astype(np.float32)
    mismatches = 0
    for s in states:
        if int(model.predict(s, deterministic=True)[0]) != act(s, weights):
            mismatches += 1
    assert mismatches == 0, f"Не совпало действий: {mismatches} из 1000"
