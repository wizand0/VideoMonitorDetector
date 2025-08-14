@echo off
REM --- Включаем поддержку UTF-8 для корректного отображения кириллицы ---
chcp 65001 >nul

echo ================================
echo 🚀 Запуск анализа движения
echo ================================

REM --- Создаём файл requirements.txt ---
REM --- echo opencv-python > requirements.txt ---
REM --- echo numpy >> requirements.txt ---
REM --- echo tqdm >> requirements.txt ---

REM --- Создаём виртуальное окружение ---
if not exist venv (
    echo 📦 Создаю виртуальное окружение...
    python -m venv venv
)

REM --- Активируем виртуальное окружение ---
call venv\Scripts\activate

REM --- Устанавливаем зависимости из requirements.txt ---
echo 📥 Устанавливаю необходимые пакеты...
pip install --upgrade pip
pip install -r requirements.txt

REM --- Запускаем основной скрипт ---
echo ▶ Запуск скрипта анализа...
python motion_detector.py

pause
