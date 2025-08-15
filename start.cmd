@echo off
chcp 65001 >nul

echo =====================================
echo 🚀 Выбор режима работы анализатора
echo =====================================
echo 1 - Анализ готовых видеофайлов
echo 2 - Анализ RTSP потоков с камер
echo =====================================

set /p mode="Введите номер режима (1 или 2): "

if "%mode%"=="1" (
    echo 📂 Режим: анализ готовых видео
    call run_motion.cmd
    exit /b
) else if "%mode%"=="2" (
    echo 📡 Режим: анализ RTSP потоков
    call rtsp_run_motion.cmd
    exit /b
) else (
    echo ❌ Неверный выбор. Завершение.
    pause
    exit /b
)
