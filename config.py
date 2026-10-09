"""Константы варианта 16 (ЛР7). Все остальные модули берут значения отсюда."""

import os

ENV_ID = "CartPole-v1"
SEED = 116
EVAL_EPISODES = 20
ACTION_MODE = "greedy"  # действие = argmax логитов; в SB3 это model.predict(obs, deterministic=True)
MODEL_VERSION = "ppo-cartpole-v1.0"
REPO_NAME = "SII-LR07-Nazdratenko"
TAG = "v1.0"
PYTHON_MIN = "3.10"
AUTHOR = "Nazdratenko"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

SB3_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_cartpole_v1.0.zip")
NUMPY_POLICY_PATH = os.path.join(MODELS_DIR, "policy_v1.0.npz")
CANDIDATE_DIR = os.path.join(MODELS_DIR, "candidate")
EVALUATION_JSON_PATH = os.path.join(RESULTS_DIR, "evaluation.json")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")
ACTIONS_LOG_PATH = os.path.join(LOGS_DIR, "actions.jsonl")

TRAIN_ENABLED = os.environ.get("TRAIN_ENABLED", "false").lower() == "true"
TRAIN_MAX_TIMESTEPS = int(os.environ.get("TRAIN_MAX_TIMESTEPS", "50000"))
TRAIN_DEFAULT_TIMESTEPS = 30000
TRAIN_MIN_TIMESTEPS = 5000
EXPECTED_MIN_MEAN_REWARD = 475.0
EXPORT_CHECK_STATES = 1000
