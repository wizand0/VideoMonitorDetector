import cv2
import numpy as np
import os
import csv
import time
import logging
import json
import simpleaudio as sa
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# ===================== НАСТРОЙКИ =====================
SENSITIVITY = 25
MIN_AREA = 800
PLAYBACK_SPEED = 8  # чем выше, тем меньше нагрузка
SAVE_FRAMES = True
SAVE_DELAY_SEC = 2
RECOGNITION_DELAY_SEC = 4
OUTPUT_FILE = "rtsp_motions_log.csv"
FRAMES_DIR = "rtsp_motion_frames"
MAX_THREADS = 4
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.7
ALARM_FILE = "icq.wav"
TARGET_CLASSES = ["person", "cat", "dog"]
ALARM_COOLDOWN = 6  # задержка перед повторным сигналом

# Окно для показа движения
DISPLAY_WINDOW_NAME = "Обнаружено движение"
DISPLAY_DURATION = 3  # секунды
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Логгер
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("motion_debug.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Подготовка директорий
os.makedirs(FRAMES_DIR, exist_ok=True)

# Глобальные переменные
last_detection = {}  # {camera_name: {"class": str, "time": float}}
last_display_time = 0
display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
SHOW_WINDOW = False
PLAY_ALARM = False

# Загрузка YOLO
logging.info("📦 Загружаю модель YOLOv8...")
model = YOLO(YOLO_MODEL)

# Подготовка CSV
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["camera", "timestamp", "class", "confidence"])


def play_alarm():
    if not PLAY_ALARM:
        return
    try:
        wave_obj = sa.WaveObject.from_wave_file(ALARM_FILE)
        wave_obj.play()
    except Exception as e:
        logging.error(f"Не удалось проиграть сигнал: {e}")


def update_display(frame):
    """Обновляет содержимое общего окна."""
    global last_display_time, display_frame
    if SHOW_WINDOW:
        resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        display_frame = resized
        last_display_time = time.time()


def display_loop():
    """Постоянно обновляет общее окно, пока идёт работа."""
    while SHOW_WINDOW:
        now = time.time()
        if now - last_display_time > DISPLAY_DURATION:
            frame_to_show = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        else:
            frame_to_show = display_frame
        cv2.imshow(DISPLAY_WINDOW_NAME, frame_to_show)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def detect_motion_and_objects(camera_name, rtsp_url):
    global last_detection
    try:
        logging.info(f"▶ Подключаюсь к камере {camera_name}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logging.error(f"❌ Не удалось подключиться к {camera_name}")
            return

        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        last_save_time = 0
        last_recognition_time = 0
        frame_count = 0

        while ret:
            if frame_count % PLAYBACK_SPEED != 0:
                frame1 = frame2
                ret, frame2 = cap.read()
                frame_count += 1
                continue

            # Анализ движения на уменьшенных кадрах для экономии CPU
            small1 = cv2.resize(frame1, (640, 360))
            small2 = cv2.resize(frame2, (640, 360))
            diff = cv2.absdiff(small1, small2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, SENSITIVITY, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = any(cv2.contourArea(c) >= MIN_AREA for c in contours)

            if motion_detected and (time.time() - last_recognition_time >= RECOGNITION_DELAY_SEC):
                last_recognition_time = time.time()

                # Запуск YOLO
                results = model(frame2, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    conf = float(box.conf[0])

                    if class_name in TARGET_CLASSES and conf >= CONF_THRESHOLD:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        logging.info(f"🧍 {class_name} ({conf:.2f}) на {camera_name} в {ts}")

                        # Анти-спам
                        prev = last_detection.get(camera_name, {})
                        if (prev.get("class") != class_name) or (time.time() - prev.get("time", 0) > ALARM_COOLDOWN):
                            play_alarm()
                            last_detection[camera_name] = {"class": class_name, "time": time.time()}

                        # Показ кадра в окне
                        update_display(frame2)

                        # Сохранение кадра
                        if SAVE_FRAMES and (time.time() - last_save_time >= SAVE_DELAY_SEC):
                            frame_file = os.path.join(FRAMES_DIR, f"{camera_name}_{ts.replace(':','-')}_{class_name}.jpg")
                            cv2.imwrite(frame_file, frame2)
                            last_save_time = time.time()

                        # Запись в CSV
                        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([camera_name, ts, class_name, f"{conf:.2f}"])
                        break

            frame1 = frame2
            ret, frame2 = cap.read()
            frame_count += 1

        cap.release()

    except Exception as e:
        logging.error(f"Ошибка при обработке {camera_name}: {e}")


def main():
    global SHOW_WINDOW, PLAY_ALARM

    # Вопросы при старте
    PLAY_ALARM = input("Проигрывать звуковое уведомление? (y/n): ").strip().lower() == "y"
    SHOW_WINDOW = input("Показывать кадры с движением? (y/n): ").strip().lower() == "y"

    # Загрузка камер
    with open("cameras.json", "r", encoding="utf-8") as f:
        cameras = json.load(f)

    # Проверка числа камер
    if len(cameras) > MAX_THREADS:
        logging.error(f"В конфиге {len(cameras)} камер, но можно обрабатывать максимум {MAX_THREADS} одновременно!")
        return

    logging.info(f"🔍 Найдено {len(cameras)} камер. Запускаю в {len(cameras)} поток(ах)...")

    # Запуск потока показа окна, если нужно
    if SHOW_WINDOW:
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DISPLAY_WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        import threading
        threading.Thread(target=display_loop, daemon=True).start()

    # Запуск камер
    with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        for name, url in cameras.items():
            executor.submit(detect_motion_and_objects, name, url)

    logging.info("✅ Работа завершена")


if __name__ == "__main__":
    main()
