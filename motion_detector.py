import cv2
import numpy as np
import os
import csv
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import YOLO

# ===================== НАСТРОЙКИ =====================
SENSITIVITY = 25
MIN_AREA = 800
SHOW_WINDOW = False
WINDOW_SCALE = 0.6
PLAYBACK_SPEED = 16
SAVE_FRAMES = True
SAVE_DELAY_SEC = 2
RECOGNITION_DELAY_SEC = 3
OUTPUT_FILE = "motions_log.csv"
FRAMES_DIR = "motion_frames"
MAX_THREADS = 6
# YOLO_MODEL = "yolov8n.pt"
YOLO_MODEL = "yolov8s.pt"
# YOLO_MODEL = "yolov8m.pt"
# YOLO_MODEL = "yolov8l.pt"
# YOLO_MODEL = "yolov8x.pt"
CONF_THRESHOLD = 0.7

# Классы для распознавания (можно указать любые из списка COCO)
# Full list is: person, bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,
# bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,
# sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,
# banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,
# laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush,


TARGET_CLASSES = ["person", "cat", "dog"]  # Пример: люди, машины, собаки
#TARGET_CLASSES = TARGET_CLASSES = list(model.names.values())

# ======================================================

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

# Загрузка YOLO
logging.info("📦 Загружаю модель YOLOv8...")
model = YOLO(YOLO_MODEL)

# Подготовка CSV
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "frame_time_sec", "class", "confidence"])

def detect_motion_and_objects(video_path, position=0):
    try:
        logging.info(f"▶ Обработка файла: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"❌ Не удалось открыть {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        last_save_time = 0
        last_recognition_time = 0
        frame_count = 0
        detections = []

        # Прогресс-бар для кадров в одном файле
        with tqdm(total=total_frames, desc=os.path.basename(video_path), position=position+1, leave=False) as pbar:
            while ret and not cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 0:
                diff = cv2.absdiff(frame1, frame2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, SENSITIVITY, 255, cv2.THRESH_BINARY)
                dilated = cv2.dilate(thresh, None, iterations=3)
                contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected = any(cv2.contourArea(c) >= MIN_AREA for c in contours)

                if motion_detected and (time.time() - last_recognition_time >= RECOGNITION_DELAY_SEC):
                    last_recognition_time = time.time()
                    results = model(frame2, verbose=False)[0]
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        class_name = results.names[cls_id]
                        conf = float(box.conf[0])

                        if class_name in TARGET_CLASSES and conf >= CONF_THRESHOLD:
                            # Рисуем рамку
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame2, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # Лог
                            ts = frame_count / fps
                            logging.info(f"🧍 {class_name} ({conf:.2f}) в {video_path}, {ts:.2f} сек")
                            detections.append((video_path, ts, class_name, conf))

                            # Сохраняем кадр
                            if SAVE_FRAMES and (time.time() - last_save_time >= SAVE_DELAY_SEC):
                                frame_file = os.path.join(FRAMES_DIR, f"{os.path.basename(video_path)}_{ts:.2f}_{class_name}.jpg")
                                cv2.imwrite(frame_file, frame2)
                                last_save_time = time.time()

                            # Запись в CSV
                            with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow([video_path, ts, class_name, f"{conf:.2f}"])

                            break  # чтобы не логгировать один и тот же объект несколько раз за кадр

                if SHOW_WINDOW:
                    resized = cv2.resize(frame2, (0, 0), fx=WINDOW_SCALE, fy=WINDOW_SCALE)
                    cv2.imshow("Video", resized)
                    if cv2.waitKey(int(1000 / (fps * PLAYBACK_SPEED))) & 0xFF == ord("q"):
                        break

                frame1 = frame2
                ret, frame2 = cap.read()
                frame_count += 1
                pbar.update(1)  # обновляем прогресс-бар

        cap.release()
        logging.info(f"✅ Завершена обработка {video_path}")
        return detections

    except Exception as e:
        logging.error(f"Ошибка при обработке {video_path}: {e}")
        return []

def main():
    video_files = [f for f in os.listdir() if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    logging.info(f"🔍 Найдено {len(video_files)} видеофайлов. Запускаю в {MAX_THREADS} поток(ах)...")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        with tqdm(total=len(video_files), desc="Обработка файлов", position=0) as main_pbar:
            futures = [executor.submit(detect_motion_and_objects, video_file, i) for i, video_file in enumerate(video_files)]
            for future in futures:
                future.result()
                main_pbar.update(1)

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()