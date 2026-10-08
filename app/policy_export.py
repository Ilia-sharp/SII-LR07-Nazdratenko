"""Экспорт весов SB3 PPO в npz для numpy-инференса.

Вызывается только при обучении (scripts/train.py и training_service).
Импорт torch здесь не нужен: функция принимает уже обученную модель.
"""

import numpy as np


def extract_weights(model) -> dict:
    """Достаёт W1, b1, W2, b2, W3, b3 из обученной модели SB3.

    Структура SB3 MlpPolicy: mlp_extractor.policy_net = [Linear, Tanh, Linear, Tanh],
    action_net = Linear(64 -> 2). Веса torch хранятся в виде (out, in), как и в numpy-инференсе.
    """
    policy_net = model.policy.mlp_extractor.policy_net
    linears = [m for m in policy_net if m.__class__.__name__ == "Linear"]
    if len(linears) != 2:
        raise RuntimeError(f"Ожидалось 2 слоя в policy_net, найдено {len(linears)}")
    act_fn = {m.__class__.__name__ for m in policy_net if m.__class__.__name__ != "Linear"}
    if act_fn != {"Tanh"}:
        raise RuntimeError(f"Ожидалась активация Tanh, найдено: {sorted(act_fn)}")

    l1, l2 = linears
    l3 = model.policy.action_net

    def to_np(t):
        return t.detach().cpu().numpy().astype(np.float64)

    return {
        "W1": to_np(l1.weight), "b1": to_np(l1.bias),
        "W2": to_np(l2.weight), "b2": to_np(l2.bias),
        "W3": to_np(l3.weight), "b3": to_np(l3.bias),
    }


def export_numpy_policy(model, path: str) -> None:
    """Сохраняет веса в npz."""
    np.savez(path, **extract_weights(model))


def verify_export(model, weights: dict, n_states: int, seed: int) -> None:
    """Сверяет argmax numpy-инференса с model.predict(deterministic=True) на n_states состояниях.

    Бросает RuntimeError при несовпадении. Это обязательная проверка экспорта.
    """
    from app.policy import act as numpy_act

    rng = np.random.default_rng(seed)
    states = rng.uniform(low=-0.05, high=0.05, size=(n_states, 4)).astype(np.float32)
    mismatches = 0
    for state in states:
        sb3_action = int(model.predict(state, deterministic=True)[0])
        np_action = numpy_act(state, weights)
        if sb3_action != np_action:
            mismatches += 1
    if mismatches:
        raise RuntimeError(
            f"Экспорт неверен: {mismatches} из {n_states} действий не совпали с SB3."
        )
