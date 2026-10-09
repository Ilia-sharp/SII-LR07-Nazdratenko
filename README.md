# ЛР7. Агент обучения с подкреплением как сервис

**Автор:** Nazdratenko
**Курс:** 4
**Дисциплина:** Системы искусственного интеллекта
**Лекция:** №9 «Обучение с подкреплением в индустрии»
**Тег:** v1.0

---

## 1. Цель работы

Показать обучение с подкреплением как компонент системы: обучить агента в симуляторе, сохранить политику, обернуть её в HTTP-сервис `POST /act` и вести журнал взаимодействий. Итог — рабочий веб-сервис с сайтом, на котором видно прогресс обучения и оценки.

## 2. Вариант

| Параметр | Значение |
|---|---|
| Среда | `CartPole-v1` (Gymnasium) |
| Seed | `116` |
| Эпизодов оценки | `20` (seed эпизода i = 116 + i) |
| Режим выбора действия | `greedy` (`model.predict(obs, deterministic=True)`) |
| Версия модели | `ppo-cartpole-v1.0` |

## 3. Ограничение: обучение только в симуляторе

Обучение выполняется **только в симуляторе** Gymnasium CartPole-v1. Реальных пользователей, объектов и данных в работе нет. Сервис не получает данных от людей и не управляет никаким реальным оборудованием.

## 4. Структура репозитория

```
SII-LR07-Nazdratenko/
  README.md
  render.yaml
  requirements.txt
  requirements-train.txt
  .gitignore
  .gitattributes
  run_all.bat
  run_all.sh
  config.py
  app/
    __init__.py
    main.py
    policy.py
    action_logger.py
    progress.py
    training_service.py
    evaluation_service.py
    policy_export.py
  scripts/
    train.py
    evaluate.py
    fill_readme.py
    demo_requests.py
  models/
    ppo_cartpole_v1.0.zip
    policy_v1.0.npz
  results/
    evaluation.json
    training_history.json
  logs/
    actions.jsonl
  static/
    index.html
    style.css
    app.js
  tests/
    conftest.py
    test_api.py
    test_policy_consistency.py
    test_logging.py
```

## 5. Быстрый запуск (всё одной командой)

Windows: двойной клик по `run_all.bat`. Linux/macOS: `./run_all.sh`.
Скрипт создаёт окружение, ставит зависимости, обучает PPO (seed 116), оценивает агента против случайного,
заполняет таблицу результатов в этом README, запускает тесты, создаёт демо-журнал, делает коммит, тег `v1.0`
и отправляет их на GitHub (и на GitVerse, если ввести адрес).

Ручной запуск по шагам описан ниже.

## 5.0. Установка и запуск по шагам

### 5.1. Обучение и оценка (локально)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt
python scripts/train.py --timesteps 100000
python scripts/evaluate.py
```

`train.py` сохраняет `models/ppo_cartpole_v1.0.zip` и `models/policy_v1.0.npz`.
`evaluate.py` работает **без переобучения**: если файла модели нет, он выводит «Сначала выполните python scripts/train.py» и завершается с кодом 1.

### 5.2. Сервис

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Откройте <http://localhost:8000> — сайт. Документация API: <http://localhost:8000/docs>.

### 5.3. Тесты

```bash
pytest
```

Тесты не переобучают модель.

## 6. Примеры запросов

**Проверка здоровья сервиса:**

```bash
curl -s http://localhost:8000/health
```

Ответ:

```json
{"status": "ok", "model_version": "ppo-cartpole-v1.0", "model_loaded": true}
```

**Запрос действия:**

```bash
curl -s -X POST http://localhost:8000/act \
  -H "Content-Type: application/json" \
  -d '{"state": [0.01, 0.2, -0.03, -0.4]}'
```

Ответ:

```json
{"action": 1, "model_version": "ppo-cartpole-v1.0"}
```

**С наградой предыдущего шага (необязательно):**

```bash
curl -s -X POST http://localhost:8000/act \
  -H "Content-Type: application/json" \
  -d '{"state": [0.01, 0.2, -0.03, -0.4], "reward": 1.0}'
```

## 7. Контракт API

| Метод | Путь | Вход | Выход | Коды ошибок |
|---|---|---|---|---|
| GET | `/health` | — | `status`, `model_version`, `model_loaded` | — |
| POST | `/act` | JSON `{"state": [4 числа], "reward": число или null}` | `{"action": 0\|1, "model_version": "ppo-cartpole-v1.0"}` | 422 (неверная длина или NaN/inf), 503 (нет весов) |
| GET | `/api/logs?limit=10` | — | `items` — последние записи журнала | 422 (limit вне 1..100) |
| GET | `/api/training-history` | — | запись реального обучения (шаги и средняя награда) для сайта | 404 (записи нет) |
| GET | `/api/progress` | — | состояние длительной операции | — |
| POST | `/api/evaluate` | — | 202, запуск оценки в фоне | 409 (задача уже идёт) |
| GET | `/api/evaluate/result` | — | содержимое `results/evaluation.json` | 404 (результата нет) |
| POST | `/api/train?timesteps=N` | — | 202, запуск обучения-кандидата | 403 (обучение отключено), 409, 422 |

Все ошибки возвращаются с полем `detail` на русском языке, без трейсбеков.

## 8. Формат logs/actions.jsonl

Каждый успешный `POST /act` добавляет одну строку JSON:

```json
{"time": "2026-10-08T12:00:00Z", "state": [0.01, 0.2, -0.03, -0.4], "action": 1, "reward": null, "model_version": "ppo-cartpole-v1.0"}
```

| Поле | Описание |
|---|---|
| `time` | время в UTC, ISO 8601, с суффиксом `Z` |
| `state` | четыре числа — состояние среды |
| `action` | выбранное действие: 0 (влево) или 1 (вправо) |
| `reward` | награда предыдущего шага или `null`, если клиент её не прислал |
| `model_version` | версия политики, которая выбрала действие |

Запись потокобезопасна. Если запись не удалась, ответ `/act` всё равно отдаётся, а ошибка пишется в stderr.

## 9. Результаты: PPO против случайного агента

<!-- RESULTS:START -->
После запуска `python scripts/evaluate.py` и `python scripts/fill_readme.py` здесь появится таблица.
<!-- RESULTS:END -->

## 10. Воспроизводимость

- Seed `116` задан в `config.py` и используется в `train.py` (`random`, `numpy`, `torch`, `env.reset`, `PPO(seed=...)`), в `evaluate.py` (seed эпизодов и `env.action_space.seed`).
- Версии зафиксированы в `requirements.txt` и `requirements-train.txt`: `gymnasium==1.0.0`, `numpy==2.1.3`, `torch==2.5.1`, `stable-baselines3==2.6.0`, Python 3.10+.
- `results/evaluation.json` коммитится. Повторный запуск `evaluate.py` с теми же весами даёт те же числа.
- Экспорт весов в numpy проверяется при обучении: на 1000 случайных состояниях argmax numpy-инференса должен совпасть с `model.predict(deterministic=True)`.

## 11. Применимость RL

CartPole — задача управления с отложенной обратной связью и нелинейной динамикой: ошибка в действии проявляется только через несколько шагов. Для этой конкретной задачи простое правило, например пропорциональный регулятор по углу, тоже удерживает шест, поэтому полный RL здесь избыточен по сложности и вычислениям. Полный RL оправдан тогда, когда динамику системы нельзя выписать явно, она нелинейна и многомерна, а политику приходится получать из взаимодействия с симулятором. Учебная цель работы — показать конвейер «обучить в симуляторе → сохранить политику → сервис → журнал», а не сам выбор алгоритма. Поэтому PPO выбран как стандартный и хорошо документированный алгоритм, а не потому, что без него задачу нельзя решить. Обучение выполняется только в симуляторе и не затрагивает реальных пользователей или объекты.

## 12. Ограничения деплоя (Render Free)

- **Засыпание.** Бесплатный сервис засыпает без активности. Первый запрос может ждать до минуты. Сайт показывает «Сервер просыпается» и опрашивает `/health`.
- **Эфемерная файловая система.** Файлы, созданные на сервере (включая `logs/actions.jsonl` и кандидатов обучения), сбрасываются при перезапуске. Журнал на сайте живёт до перезапуска сервиса.
- **Ресурсы.** 512 MB RAM и 0.1 CPU. Поэтому рантайм-сервис работает на numpy и не загружает torch. Обучение на сайте отключено по умолчанию (`TRAIN_ENABLED=false` в `render.yaml`).

## 13. Деплой на Render

1. Загрузите репозиторий на GitHub (и на GitVerse для сдачи).
2. В Render выберите **New → Blueprint** и укажите репозиторий.
3. Render прочитает `render.yaml` и создаст сервис `nazdratenko-cartpole-lr7` на тарифе Free.
4. Дождитесь сборки. Проверьте `/health`: должно прийти `"status": "ok"`.

Для сборки используется `requirements.txt`, в котором нет torch, поэтому сборка укладывается в ресурсы Free.

**Ссылка на сайт после деплоя:** *укажите после деплоя* — https://________.onrender.com

## 14. Допущения

<!-- ASSUMPTION:START -->
- Веса модели и результаты создаются запуском `run_all.bat` (или `run_all.sh`).
<!-- ASSUMPTION:END -->
- **Блок «Обучение» на сайте** при `TRAIN_ENABLED=false` проигрывает с анимацией запись реального обучения из `results/training_history.json`, которую создаёт `scripts/train.py`. Подпись на странице прямо говорит, что это запись, а не обучение на сервере.
- **Обучение на Render отключено** (`TRAIN_ENABLED=false`): torch не помещается в 512 MB. Кнопка на сайте неактивна с пояснением.
- **Версия Python.** Проект проверяется на Python 3.10–3.13. Для Render в `render.yaml` указано `PYTHON_VERSION=3.11.9`.
- **Значения по умолчанию на сайте:** обучение-кандидат — 30000 шагов, оценка — 20 эпизодов, seed 116.
- **Логотип.** Текстовый вордмарк «Nazdratenko» и простая SVG-иконка «шест на тележке».
- **Журнал на сайте** обновляется каждые 3 секунды и показывает 10 последних записей.
