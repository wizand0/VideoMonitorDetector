import cv2
import numpy as np
import os
import csv
import time
import logging
import json
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from ultralytics import YOLO

# ===================== НАСТРОЙКИ =====================
SENSITIVITY = 25
MIN_AREA = 800
PLAYBACK_SPEED = 8  # чем выше, тем меньше нагрузка
SAVE_FRAMES = True
RECOGNITION_DELAY_SEC = 4
OUTPUT_FILE = "rtsp_motions_log.csv"
FRAMES_DIR = "rtsp_motion_frames"
MAX_THREADS = 4
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.7
ALARM_FILE = "icq.wav"
TARGET_CLASSES = ["person", "cat", "dog"]

# Логгер
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("motion_ui_debug.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Подготовка директорий
os.makedirs(FRAMES_DIR, exist_ok=True)

# ===================== Глобальные флаги и состояние =====================
PLAY_ALARM = False
VIEW_ENABLED = True
OP_MODE = "Primed"  # Idle / Primed
_last_trigger_mark_until = 0.0
_state_lock = threading.Lock()

_view_queue: "Queue[np.ndarray]" = Queue(maxsize=8)
_view_stop = threading.Event()

# ===================== Аудио =====================
if platform.system() == "Linux":
    from playsound import playsound
    def play_alarm():
        if not PLAY_ALARM:
            return
        try:
            playsound(ALARM_FILE, block=False)
        except Exception as e:
            logging.error(f"Не удалось проиграть сигнал (Linux): {e}")
else:
    import simpleaudio as sa
    def play_alarm():
        if not PLAY_ALARM:
            return
        try:
            sa.WaveObject.from_wave_file(ALARM_FILE).play()
        except Exception as e:
            logging.error(f"Не удалось проиграть сигнал (Win): {e}")

# ===================== Утилиты =====================
def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def date_dir():
    d = time.strftime("%Y%m%d")
    p = os.path.join(FRAMES_DIR, d)
    os.makedirs(p, exist_ok=True)
    return p

def _status_line():
    with _state_lock:
        mode = OP_MODE
        alarm = "ON" if PLAY_ALARM else "OFF"
        view = "ON" if VIEW_ENABLED else "OFF"
        trig = "TRIGGERED" if time.time() < _last_trigger_mark_until else ""
    parts = [f"Mode: {mode}", f"Alarm: {alarm}", f"Display: {view}"]
    if trig:
        parts.append(trig)
    return " | ".join(parts)

def _toggle_mode():
    global OP_MODE
    with _state_lock:
        OP_MODE = "Idle" if OP_MODE == "Primed" else "Primed"
        return OP_MODE

def _toggle_alarm():
    global PLAY_ALARM
    with _state_lock:
        PLAY_ALARM = not PLAY_ALARM
        return PLAY_ALARM

def _toggle_view():
    global VIEW_ENABLED
    with _state_lock:
        VIEW_ENABLED = not VIEW_ENABLED
        return VIEW_ENABLED

def _mark_triggered(duration_sec=2.0):
    global _last_trigger_mark_until
    with _state_lock:
        _last_trigger_mark_until = time.time() + duration_sec

# ===================== UI‑поток =====================
def viewer_thread():
    cv2.namedWindow("VIEWER", cv2.WINDOW_NORMAL)
    last_img = None
    last_time = 0.0
    hold_sec = 2.0

    while not _view_stop.is_set():
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            _view_stop.set()
            break
        elif key == 32:
            logging.info(f"🔄 Mode → {_toggle_mode()}")
        elif key in (ord('a'), ord('A')):
            logging.info(f"🔔 Alarm {'ON' if _toggle_alarm() else 'OFF'}")
        elif key in (ord('w'), ord('W')):
            logging.info(f"🖼️ Display {'ON' if _toggle_view() else 'OFF'}")

        try:
            img = _view_queue.get(timeout=0.02)
            last_img = img
            last_time = time.time()
        except Empty:
            pass

        if last_img is not None and VIEW_ENABLED:
            canvas = last_img.copy()
            cv2.putText(canvas, _status_line(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("VIEWER", canvas)
            if time.time() - last_time > hold_sec:
                last_img = None
        else:
            blank = np.zeros((320, 560, 3), dtype=np.uint8)
            msg = "Display OFF" if not VIEW_ENABLED else "Waiting for events..."
            cv2.putText(blank, msg, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(blank, _status_line(), (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 230, 20), 2)
            cv2.imshow("VIEWER", blank)

    cv2.destroyWindow("VIEWER")

def show_frame(frame, camera_name, label=None):
    out = frame.copy()
    text = f"{camera_name} | {label or ''} | {now_ts()}"
    cv2.putText(out, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 20), 2)
    try:
        _view_queue.put_nowait(out)
    except Exception:
        pass

# ===================== YOLO =====================
logging.info("📦 Загружаю модель YOLOv8...")
model = YOLO(YOLO_MODEL)

if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["camera", "timestamp", "class", "confidence"])

# ===================== ДЕТЕКТОР =====================
def detect_motion_and_objects(camera_name, rtsp_url):
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logging.error(f"❌ Не удалось подключиться к {camera_name}")
            return

        ret, frame1 = cap.read()
        ret2, frame2 = cap.read()
        if not ret or not ret2:
            cap.release()
            return

        last_trigger_time = 0.0
        frame_count = 0

        while True:
            if frame_count % PLAYBACK_SPEED != 0:
                frame1 = frame2
                if not cap.grab():
                    break
                ok, frame2 = cap.retrieve()
                if not ok:
                    break
                frame_count += 1
                continue

            small1 = cv2.resize(frame1, (640, 360))
            small2 = cv2.resize(frame2, (640, 360))
            diff = cv2.absdiff(small1, small2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, SENSITIVITY, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(c) >= MIN_AREA for c in contours)

            if motion_detected:
                results = model(frame2, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    conf = float(box.conf[0])

                    if class_name in TARGET_CLASSES and conf >= CONF_THRESHOLD:
                        now = time.time()
                        if OP_MODE == "Primed" and (now - last_trigger_time) >= RECOGNITION_DELAY_SEC:
                            ts = now_ts()
                            logging.info(f"✅ {camera_name}: {class_name} ({conf:.2f}), {ts}")
                            play_alarm()
                            _mark_triggered()
                            show_frame(frame2, camera_name, class_name)

                            if SAVE_FRAMES:
                                fname = f"{camera_name}_{ts.replace(':', '-')}_{class_name}.jpg"
                                cv2.imwrite(os.path.join(date_dir(), fname), frame2)

                            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([camera_name, ts, class_name, f"{conf:.2f}"])

                            last_trigger_time = now
                        break

            frame1 = frame2
            if not cap.grab():
                break
            ok, frame2 = cap.retrieve()
            if not ok:
                break
            frame_count += 1

        cap.release()

    except Exception as e:
        logging.exception(f"Ошибка при обработке {camera_name}: {e}")

# ===================== MAIN =====================
def main():
    global PLAY_ALARM, VIEW_ENABLED

    with open("cameras.json", "r", encoding="utf-8") as c:
        cameras = json.load(c)
    if not cameras:
        logging.error("❌ cameras.json пустой.")
        return

    try:
        PLAY_ALARM = input("Проигрывать звуковое уведомление? (y/n): ").strip().lower() == "y"
    except Exception:
        PLAY_ALARM = False
    try:
        VIEW_ENABLED = input("Показывать кадры при срабатывании? (y/n): ").strip().lower() == "y"
    except Exception:
        VIEW_ENABLED = True

    ui_thread = None
    if VIEW_ENABLED:
        _view_stop.clear()
        ui_thread = threading.Thread(target=viewer_thread, daemon=True)
        ui_thread.start()

    logging.info("🔥 Горячие клавиши: Q=выход | SPACE=Idle/Primed | A=звук ON/OFF | W=показ ON/OFF")
    logging.info(f"🔍 Камер: {len(cameras)}. Запускаю в {min(len(cameras), MAX_THREADS)} поток(ах)…")

    try:
        with ThreadPoolExecutor(max_workers=min(len(cameras), MAX_THREADS)) as executor:
            futures = [executor.submit(detect_motion_and_objects, name, url) for name, url in cameras.items()]
            for g in futures:
                g.result()
    finally:
        _view_stop.set()
        try:
            if ui_thread:
                ui_thread.join(timeout=2)
        except Exception:
            pass
        cv2.destroyAllWindows()

    logging.info("✅ Работа завершена")

if __name__ == "__main__":
    main()
