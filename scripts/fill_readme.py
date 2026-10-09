"""Подставляет результаты из results/evaluation.json в README.md (между маркерами)."""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import EVALUATION_JSON_PATH, EXPECTED_MIN_MEAN_REWARD  # noqa: E402

README = os.path.join(ROOT, "README.md")


def replace_block(text: str, name: str, new_body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"В README нет маркеров {name}")
    return pattern.sub(lambda m: f"{m.group(1)}\n{new_body}\n{m.group(3)}", text)


def main() -> int:
    if not os.path.exists(EVALUATION_JSON_PATH):
        print("Нет results/evaluation.json. Сначала выполните python scripts/evaluate.py", file=sys.stderr)
        return 1
    with open(EVALUATION_JSON_PATH, encoding="utf-8") as fh:
        r = json.load(fh)

    ppo, rnd = r["ppo"], r["random"]
    if ppo["mean"] < EXPECTED_MIN_MEAN_REWARD:
        print(f"Средняя награда PPO {ppo['mean']} ниже {EXPECTED_MIN_MEAN_REWARD}: модель недообучена, README не изменён.", file=sys.stderr)
        return 3
    table = (
        "| Агент | Среднее | std | min | max | Эпизодов |\n"
        "|---|---|---|---|---|---|\n"
        f"| PPO (greedy) | {ppo['mean']:.2f} | {ppo['std']:.2f} | {ppo['min']} | {ppo['max']} | {r['episodes']} |\n"
        f"| Случайный агент | {rnd['mean']:.2f} | {rnd['std']:.2f} | {rnd['min']} | {rnd['max']} | {r['episodes']} |\n\n"
        f"Seed {r['seed']}, режим `{r['action_mode']}`, модель `{r['model_version']}`, дата расчёта {r['date']}.\n\n"
        f"**Вывод:** обученный PPO в среднем держит шест {ppo['mean']:.1f} шагов из 500 возможных, "
        f"случайный агент — {rnd['mean']:.1f}. PPO лучше случайного агента в {r['improvement_ratio']} раза. "
        "Эпизоды и seed одинаковы для обоих агентов, поэтому сравнение честное."
    )
    assumption = (
        "- **Веса модели и результаты.** `models/ppo_cartpole_v1.0.zip`, `models/policy_v1.0.npz` и "
        "`results/evaluation.json` получены запуском `scripts/train.py` и `scripts/evaluate.py` "
        "(seed 116, 100000 шагов) и закоммичены в репозиторий."
    )

    with open(README, encoding="utf-8") as fh:
        text = fh.read()
    text = replace_block(text, "RESULTS", table)
    text = replace_block(text, "ASSUMPTION", assumption)
    with open(README, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("README.md обновлён результатами оценки.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
