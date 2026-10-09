@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set GH_URL=https://github.com/Ilia-sharp/SII-LR07-Nazdratenko.git

echo ===== 1/6 Проверка Python и Git =====
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" || (echo Нужен Python 3.10 или новее: https://www.python.org/downloads/ & goto :fail)
git --version >nul 2>&1 || (echo Нужен Git: https://git-scm.com/downloads & goto :fail)

echo ===== 2/6 Окружение и зависимости (первый раз 5-15 минут: скачивается torch) =====
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-train.txt || goto :fail

echo ===== 3/6 Обучение PPO (seed 116) =====
python scripts\train.py --timesteps 100000 || goto :fail
python scripts\evaluate.py || goto :fail
python scripts\fill_readme.py
if errorlevel 1 (
  echo Качество ниже ожидаемого. Дообучаю на 300000 шагов...
  python scripts\train.py --timesteps 300000 || goto :fail
  python scripts\evaluate.py || goto :fail
  python scripts\fill_readme.py || goto :fail
)

echo ===== 4/6 Тесты и демо-журнал =====
python -m pytest -q || goto :fail
python scripts\demo_requests.py || goto :fail

echo ===== 5/6 Коммит и тег v1.0 =====
git config user.name >nul 2>&1 || git config user.name "Ilia-sharp"
git config user.email >nul 2>&1 || git config user.email "trash9305@gmail.com"
git remote get-url origin >nul 2>&1 || git remote add origin %GH_URL%
git branch -M main
git add -A
git commit -m "ЛР7: обученная модель и результаты"
git tag -fa v1.0 -m "ЛР7 v1.0"

echo ===== 6/6 Отправка на GitHub =====
git push -u origin main || goto :pushfail
git push -f origin v1.0 || goto :pushfail

echo.
set /p GV=Адрес репозитория GitVerse (Enter - пропустить): 
if not "%GV%"=="" (
  git remote remove gitverse >nul 2>&1
  git remote add gitverse %GV%
  git push -u gitverse main
  git push -f gitverse v1.0
)
echo.
echo ГОТОВО. Код и тег v1.0 отправлены. Следующий шаг: Render - New - Blueprint.
pause
exit /b 0

:pushfail
echo.
echo Push не удался. Скорее всего, не выполнен вход в GitHub (логин через браузер или токен).
echo Все файлы уже закоммичены. Повторите: git push -u origin main ^&^& git push -f origin v1.0
pause
exit /b 1

:fail
echo.
echo ОШИБКА на одном из шагов выше. Пришлите текст ошибки.
pause
exit /b 1
