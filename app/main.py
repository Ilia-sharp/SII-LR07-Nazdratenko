"""FastAPI-сервис: POST /act, GET /health, служебные эндпоинты для сайта.

Рантайм не импортирует torch и stable_baselines3 (требование тарифа 512 MB RAM).
"""

import logging
import math
import os
import sys
import threading
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import action_logger, policy, progress
from app.evaluation_service import evaluate_agents, load_result, save_result
from config import (
    MODEL_VERSION,
    TRAINING_HISTORY_PATH,
    STATIC_DIR,
    TRAIN_DEFAULT_TIMESTEPS,
    TRAIN_ENABLED,
    TRAIN_MAX_TIMESTEPS,
    TRAIN_MIN_TIMESTEPS,
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cartpole")

app = FastAPI(title="ЛР7. Агент CartPole как сервис", version=MODEL_VERSION)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- модели запроса/ответа ----------

class ActRequest(BaseModel):
    state: list[float] = Field(..., description="Четыре числа: позиция, скорость, угол, угловая скорость")
    reward: Optional[float] = Field(default=None, description="Награда предыдущего шага (необязательно)")

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("Поле state должно содержать ровно 4 числа")
        if not all(math.isfinite(x) for x in value):
            raise ValueError("Все значения state должны быть конечными числами (без NaN и inf)")
        return value

    @field_validator("reward")
    @classmethod
    def _check_reward(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("reward должен быть конечным числом или null")
        return value


class ActResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    action: int
    model_version: str


# ---------- обработка ошибок: русский текст, без трейсбеков ----------

@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        msg = err.get("msg", "")
        # pydantic оборачивает наше сообщение в "Value error, ..."
        msg = msg.replace("Value error, ", "")
        messages.append(msg)
    text = "; ".join(messages) or "Некорректный запрос"
    return JSONResponse(status_code=422, content={"detail": text})


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    log.exception("Внутренняя ошибка: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера. Попробуйте позже."})


# ---------- контракт API ----------

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    try:
        policy.load_weights()
        loaded = True
    except policy.PolicyNotLoadedError:
        loaded = False
    return {"status": "ok", "model_version": MODEL_VERSION, "model_loaded": loaded}


@app.post("/act", response_model=ActResponse)
def act(body: ActRequest):
    try:
        weights = policy.load_weights()
    except policy.PolicyNotLoadedError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    action = policy.act(body.state, weights)
    record = action_logger.build_record(body.state, action, body.reward, MODEL_VERSION)
    action_logger.append(record)
    return {"action": action, "model_version": MODEL_VERSION}


# ---------- служебные эндпоинты для сайта ----------

@app.get("/api/status")
def api_status():
    return {"model_version": MODEL_VERSION, "train_enabled": TRAIN_ENABLED,
            "train_max_timesteps": TRAIN_MAX_TIMESTEPS,
            "model_loaded": policy.is_loaded() or _try_load()}


def _try_load() -> bool:
    try:
        policy.load_weights()
        return True
    except policy.PolicyNotLoadedError:
        return False


@app.get("/api/logs")
def api_logs(limit: int = Query(default=10, ge=1, le=100)):
    return {"items": action_logger.read_last(limit)}


@app.get("/api/training-history")
def api_training_history():
    """Запись реального обучения (из scripts/train.py) для воспроизведения на сайте."""
    if not os.path.exists(TRAINING_HISTORY_PATH):
        return JSONResponse(status_code=404, content={"detail": "Записи обучения пока нет. Выполните scripts/train.py."})
    import json
    with open(TRAINING_HISTORY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/api/progress")
def api_progress():
    return progress.snapshot()


def _background(job_type: str, total: int, target):
    def runner():
        try:
            result = target()
            progress.finish(result)
        except Exception as exc:  # noqa: BLE001 — в progress уходит только текст
            log.exception("Фоновая задача %s завершилась ошибкой", job_type)
            progress.fail(f"Ошибка: {exc}" if str(exc) else "Операция завершилась ошибкой")
    threading.Thread(target=runner, daemon=True).start()


@app.post("/api/evaluate", status_code=202)
def api_evaluate():
    try:
        progress.start_job("evaluate", total=40, stage="evaluate")
    except progress.JobBusyError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def job():
        weights = policy.load_weights()

        def ppo_fn(obs):
            return policy.act(obs, weights)

        def cb(done, total):
            progress.update(done=done, stage="evaluate", message=f"Эпизод {done} из {total}")

        result = evaluate_agents(ppo_fn, progress_callback=cb)
        save_result(result)
        return {"ppo_mean": result["ppo"]["mean"], "random_mean": result["random"]["mean"]}

    _background("evaluate", 40, job)
    return {"job": "evaluate", "status": "started"}


@app.get("/api/evaluate/result")
def api_evaluate_result():
    result = load_result()
    if result is None:
        return JSONResponse(status_code=404, content={"detail": "Результата оценки пока нет. Запустите оценку."})
    return result


@app.post("/api/train")
def api_train(timesteps: int = Query(default=TRAIN_DEFAULT_TIMESTEPS)):
    if not TRAIN_ENABLED:
        return JSONResponse(status_code=403, content={
            "detail": "Обучение на этом сервере отключено, модель обучена заранее в симуляторе."})
    if not (TRAIN_MIN_TIMESTEPS <= timesteps <= min(TRAIN_MAX_TIMESTEPS, 50000)):
        return JSONResponse(status_code=422, content={
            "detail": f"Число шагов должно быть от {TRAIN_MIN_TIMESTEPS} до {min(TRAIN_MAX_TIMESTEPS, 50000)}"})
    try:
        progress.start_job("train", total=timesteps, stage="init")
    except progress.JobBusyError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    from app.training_service import run_training_candidate
    _background("train", timesteps, lambda: run_training_candidate(timesteps))
    return {"job": "train", "status": "started", "timesteps": timesteps}
