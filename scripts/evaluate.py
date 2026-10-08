"""Оценка сохранённой политики PPO против случайного агента. БЕЗ переобучения.

Запуск из корня проекта:
    python scripts/evaluate.py

Результат печатается в консоль и сохраняется в results/evaluation.json.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import ENV_ID, SB3_MODEL_PATH, SEED  # noqa: E402


def main() -> int:
    if not os.path.exists(SB3_MODEL_PATH):
        print("Сначала выполните python scripts/train.py", file=sys.stderr)
        return 1

    from stable_baselines3 import PPO

    from app.evaluation_service import evaluate_agents, print_table, save_result
    from config import EVALUATION_JSON_PATH

    model = PPO.load(SB3_MODEL_PATH, device="cpu")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    print(f"Оценка: среда {ENV_ID}, seed={SEED}, 20 эпизодов, режим greedy")
    result = evaluate_agents(ppo_policy)
    print_table(result)
    save_result(result)
    print(f"\nРезультат сохранён: {EVALUATION_JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
