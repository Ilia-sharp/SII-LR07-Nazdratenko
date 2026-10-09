#!/usr/bin/env bash
# Полный цикл: окружение, обучение, оценка, тесты, README, коммит, тег, push.
set -e
cd "$(dirname "$0")"
GH_URL=https://github.com/Ilia-sharp/SII-LR07-Nazdratenko.git

echo "===== 1/6 Проверка Python и Git ====="
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" || { echo "Нужен Python 3.10+"; exit 1; }
command -v git >/dev/null || { echo "Нужен Git"; exit 1; }

echo "===== 2/6 Окружение и зависимости (первый раз 5-15 минут: скачивается torch) ====="
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-train.txt

echo "===== 3/6 Обучение PPO (seed 116) ====="
python scripts/train.py --timesteps 100000
python scripts/evaluate.py
if ! python scripts/fill_readme.py; then
  echo "Качество ниже ожидаемого. Дообучаю на 300000 шагов..."
  python scripts/train.py --timesteps 300000
  python scripts/evaluate.py
  python scripts/fill_readme.py
fi

echo "===== 4/6 Тесты и демо-журнал ====="
python -m pytest -q
python scripts/demo_requests.py

echo "===== 5/6 Коммит и тег v1.0 ====="
git config user.name >/dev/null || git config user.name "Ilia-sharp"
git config user.email >/dev/null || git config user.email "trash9305@gmail.com"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$GH_URL"
git branch -M main
git add -A
git commit -m "ЛР7: обученная модель и результаты"
git tag -fa v1.0 -m "ЛР7 v1.0"

echo "===== 6/6 Отправка на GitHub ====="
git push -u origin main && git push -f origin v1.0 || {
  echo "Push не удался: выполните вход в GitHub (браузер или токен). Файлы уже закоммичены."
  echo "Повторите: git push -u origin main && git push -f origin v1.0"; exit 1; }

read -r -p "Адрес репозитория GitVerse (Enter - пропустить): " GV
if [ -n "$GV" ]; then
  git remote remove gitverse 2>/dev/null || true
  git remote add gitverse "$GV"
  git push -u gitverse main && git push -f gitverse v1.0
fi
echo "ГОТОВО. Код и тег v1.0 отправлены. Следующий шаг: Render - New - Blueprint."
