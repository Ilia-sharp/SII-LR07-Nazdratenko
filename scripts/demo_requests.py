"""Создаёт демонстрационные записи в logs/actions.jsonl.

Играет короткий реальный эпизод CartPole-v1 (seed 116): состояние отправляется в POST /act,
выбранное действие применяется в среде, награда уходит в следующем запросе.
Запускать после обучения: python scripts/demo_requests.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402

STEPS = 8


def main() -> int:
    if not os.path.exists(config.NUMPY_POLICY_PATH):
        print("Нет models/policy_v1.0.npz. Сначала выполните python scripts/train.py", file=sys.stderr)
        return 1

    import gymnasium as gym
    from fastapi.testclient import TestClient

    from app.main import app

    if os.path.exists(config.ACTIONS_LOG_PATH):
        os.remove(config.ACTIONS_LOG_PATH)  # журнал демо начинается с чистого листа

    client = TestClient(app)
    env = gym.make(config.ENV_ID)
    obs, _ = env.reset(seed=config.SEED)
    reward = None
    for _ in range(STEPS):
        body = {"state": [float(x) for x in obs], "reward": reward}
        resp = client.post("/act", json=body)
        resp.raise_for_status()
        action = resp.json()["action"]
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    print(f"Записано {STEPS} демо-записей в {config.ACTIONS_LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
