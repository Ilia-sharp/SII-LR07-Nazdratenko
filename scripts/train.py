"""Обучение PPO в симуляторе CartPole-v1 и экспорт весов.

Запуск из корня проекта:
    python scripts/train.py --timesteps 100000

Обучение выполняется ТОЛЬКО в симуляторе Gymnasium. Реальных пользователей и объектов нет.
"""

import argparse
import os
import random
import sys

# Чтобы импорт config и app работал при запуске из любого каталога
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    ENV_ID,
    EXPECTED_MIN_MEAN_REWARD,
    EXPORT_CHECK_STATES,
    MODELS_DIR,
    MODEL_VERSION,
    NUMPY_POLICY_PATH,
    SB3_MODEL_PATH,
    SEED,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучение PPO на CartPole-v1 (ЛР7, вариант 16)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Число шагов обучения")
    parser.add_argument("--out", type=str, default=MODELS_DIR, help="Каталог для весов")
    args = parser.parse_args()

    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from app import progress
    from app.evaluation_service import _run_episode
    from app.policy_export import export_numpy_policy, extract_weights, verify_export
    from app.policy import act as numpy_act  # noqa: F401 — проверяем, что модуль импортируется

    # Воспроизводимость
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # noqa: BLE001 — не все операции поддерживают детерминизм
        pass

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, os.path.basename(SB3_MODEL_PATH))
    npz_path = os.path.join(args.out, os.path.basename(NUMPY_POLICY_PATH))

    class PrintProgressCallback(BaseCallback):
        def __init__(self, total: int):
            super().__init__()
            self.total = total

        def _on_step(self) -> bool:
            if self.num_timesteps % 1000 == 0 or self.num_timesteps >= self.total:
                ep = self.model.ep_info_buffer
                mean_r = (sum(e["r"] for e in ep) / len(ep)) if len(ep) > 0 else None
                percent = 100.0 * min(self.num_timesteps, self.total) / self.total
                progress.update(done=min(self.num_timesteps, self.total), stage="train",
                                mean_reward=mean_r)
                mr = f"{mean_r:.1f}" if mean_r is not None else "—"
                print(f"[train] шаг {self.num_timesteps}/{self.total} ({percent:.0f}%), "
                      f"средняя награда: {mr}", flush=True)
            return True

    env = gym.make(ENV_ID)
    env.reset(seed=SEED)
    model = PPO("MlpPolicy", env, seed=SEED, device="cpu", verbose=0)

    print(f"Обучение PPO: {args.timesteps} шагов, seed={SEED}, среда {ENV_ID}")
    model.learn(total_timesteps=args.timesteps, callback=PrintProgressCallback(args.timesteps))

    model.save(zip_path)
    print(f"Модель SB3 сохранена: {zip_path}")

    # Экспорт в numpy и обязательная проверка совпадения действий
    export_numpy_policy(model, npz_path)
    weights = extract_weights(model)
    try:
        verify_export(model, weights, n_states=EXPORT_CHECK_STATES, seed=SEED)
    except RuntimeError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2
    print(f"Веса numpy сохранены и проверены на {EXPORT_CHECK_STATES} состояниях: {npz_path}")

    # Быстрая проверка качества (20 эпизодов, greedy)
    rewards = []
    eval_env = gym.make(ENV_ID)
    for i in range(20):
        rewards.append(_run_episode(eval_env, lambda obs: int(model.predict(obs, deterministic=True)[0]), SEED + i))
    eval_env.close()
    mean_reward = float(np.mean(rewards))
    print(f"Средняя награда greedy на 20 эпизодах: {mean_reward:.1f} (ожидание: ≥ {EXPECTED_MIN_MEAN_REWARD})")
    if mean_reward < EXPECTED_MIN_MEAN_REWARD:
        print("ПРЕДУПРЕЖДЕНИЕ: качество ниже ожидаемого. Попробуйте увеличить --timesteps "
              "(например, 200000).")

    print(f"Готово. Версия модели: {MODEL_VERSION}. Запустите: python scripts/evaluate.py")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
