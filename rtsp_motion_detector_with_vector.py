"""
rtsp_motion_detector_with_vector.py
Реагирует только если человек прошёл из зоны START в зону FINISH.
Зоны настраиваются клавишами в одном окне (без мыши).
Есть безопасный показ последнего срабатывания на 2 секунды (отдельный UI-поток).
"""

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
from queue import Queue, Empty
import threading

# ===================== НАСТРОЙКИ =====================
SENSITIVITY = 25                  # порог бинаризации разности
MIN_AREA = 800                    # минимальная площадь контура движения
PLAYBACK_SPEED = 8                # чем выше, тем меньше нагрузка (обрабатываем каждый N-й кадр)
SAVE_FRAMES = True
RECOGNITION_DELAY_SEC = 4         # минимум секунд между триггерами/сохранениями
DIRECTION_TIMEOUT_SEC = 8         # сколько секунд ждём FINISH после входа в START
OUTPUT_FILE = "rtsp_motions_log.csv"
FRAMES_DIR = "rtsp_motion_frames"
ZONES_FILE = "zones.json"
MAX_THREADS = 4
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.7
ALARM_FILE = "icq.wav"
TARGET_CLASSES = ["person"]       # добавь "cat","dog" при необходимости

# Предпросмотр зон при старте
PREVIEW_ZONES = True
PREVIEW_ZONES_SECONDS = 2

# UI окно (ASCII-only, чтобы не ломать обработчики на Windows)
SETUP_WINDOW = "ZONE_SETUP"
VIEW_WINDOW = "VIEWER"

# Логгер
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("motion_vector_debug.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Подготовка директорий
os.makedirs(FRAMES_DIR, exist_ok=True)

# Глобальные флаги
PLAY_ALARM = False
SHOW_WINDOW = True  # включаем UI-поток показа срабатываний

# <<< Глобальное состояние и блокировка для потокобезопасности
_state_lock = threading.Lock()
OP_MODE = "Primed"     # "Primed" — работает; "Idle" — пауза триггеров
VIEW_ENABLED = True    # показ кадров включён/выключен горячей клавишей
_last_trigger_mark_until = 0.0  # до какого времени писать "TRIGGERED" в статусе

def _set_mode(new_mode: str):
    global OP_MODE
    with _state_lock:
        OP_MODE = new_mode

def _get_mode() -> str:
    with _state_lock:
        return OP_MODE

def _toggle_mode():
    with _state_lock:
        global OP_MODE
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

# YOLO
logging.info("📦 Загружаю модель YOLOv8...")
model = YOLO(YOLO_MODEL)

# CSV заголовок
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["camera", "timestamp", "class", "confidence"])

# ===================== УТИЛИТЫ =====================
def play_alarm():
    if not PLAY_ALARM:
        return
    try:
        sa.WaveObject.from_wave_file(ALARM_FILE).play()
    except Exception as e:
        logging.error(f"Не удалось проиграть сигнал: {e}")

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def date_dir() -> str:
    d = time.strftime("%Y%m%d")
    p = os.path.join(FRAMES_DIR, d)
    os.makedirs(p, exist_ok=True)
    return p

def clamp_rect(x1, y1, x2, y2, w, h):
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    x1 = max(0, min(x1, w - 1));  x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1));  y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2

def in_zone(point, rect):
    if not rect or len(rect) != 4:
        return False
    cx, cy = point
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    return x1 <= cx <= x2 and y1 <= cy <= y2

# ===================== ОКНО-ПРОСМОТРЩИК (UI-поток) =====================
_view_queue: "Queue[np.ndarray]" = Queue(maxsize=8)
_view_stop = threading.Event()

def viewer_thread():
    cv2.namedWindow(VIEW_WINDOW, cv2.WINDOW_NORMAL)
    last_img = None
    last_time = 0.0
    hold_sec = 2.0  # показываем кадр 2 секунды

    while not _view_stop.is_set():
        # гор. клавиши читаем тут же
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            _view_stop.set()
            break
        elif key == 32:  # SPACE — toggle mode Idle/Primed
            new_mode = _toggle_mode()
            logging.info(f"🔄 Mode → {new_mode}")
        elif key in (ord('a'), ord('A')):  # A — toggle alarm
            s = _toggle_alarm()
            logging.info(f"🔔 Alarm {'ON' if s else 'OFF'}")
        elif key in (ord('w'), ord('W')):  # W — toggle display
            s = _toggle_view()
            logging.info(f"🖼️ Display {'ON' if s else 'OFF'}")
            # не закрываем окно, а просто гасим вывод — так hotkeys продолжают работать

        # забираем новый кадр если есть
        try:
            img = _view_queue.get(timeout=0.02)
            last_img = img
            last_time = time.time()
        except Empty:
            pass

        # подготовка холста
        if last_img is not None and VIEW_ENABLED:
            canvas = last_img.copy()
            # наложим строку статуса поверх присланного кадра
            cv2.putText(canvas, _status_line(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 20), 2)
            cv2.imshow(VIEW_WINDOW, canvas)
            # сброс через hold_sec
            if time.time() - last_time > hold_sec:
                last_img = None
        else:
            blank = np.zeros((320, 560, 3), dtype=np.uint8)
            msg = "Display OFF" if not VIEW_ENABLED else "Waiting for events..."
            cv2.putText(blank, f"{msg}", (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            cv2.putText(
                blank,
                "Hotkeys: Q=quit | SPACE=Idle/Primed | A=alarm on/off | W=display on/off",
                (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1
            )
            # текущий статус
            cv2.putText(blank, _status_line(), (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 20), 2)
            cv2.imshow(VIEW_WINDOW, blank)

    cv2.destroyWindow(VIEW_WINDOW)

def show_frame(frame, camera_name, zones=None, center=None, label=None):
    """Отправить кадр в окно просмотра (UI-поток сам покажет его ~2 сек)."""
    out = frame.copy()
    if zones:
        z = zones.get("start")
        if z: cv2.rectangle(out, (z[0], z[1]), (z[2], z[3]), (0, 180, 0), 2)
        z = zones.get("finish")
        if z: cv2.rectangle(out, (z[0], z[1]), (z[2], z[3]), (0, 0, 255), 2)
    if center:
        cv2.circle(out, center, 6, (255, 255, 0), -1)
    # верхняя строка
    text = f"{camera_name} | {label or ''} | {now_ts()}"
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 20), 2)
    # дополнительная строка статуса
    cv2.putText(out, _status_line(), (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # не блокируемся на полной очереди
    try:
        _view_queue.put_nowait(out)
    except:
        pass

# ===================== НАСТРОЙКА ЗОН (КЛАВИШАМИ) =====================
def select_zone_keyboard(frame, title, max_w=1280, max_h=720):
    """
    Клавиатурный мастер прямоугольника в одном окне SETUP_WINDOW.
    Управление:
      W/S/A/D — сдвиг прямоугольника (10px)
      I/K — увеличить/уменьшить высоту (10px)
      L/J — увеличить/уменьшить ширину (10px)
      R — сброс (прямоугольник ~40% по центру)
      Enter — подтвердить, Esc — пропустить
    Возвращает [x1,y1,x2,y2] или None.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_w / float(w), max_h / float(h))
    disp = frame if scale == 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)))

    # начальный прямоугольник — 40% кадра по центру
    cw, ch = int(disp.shape[1] * 0.4), int(disp.shape[0] * 0.4)
    cx, cy = disp.shape[1] // 2, disp.shape[0] // 2
    x1, y1 = cx - cw // 2, cy - ch // 2
    x2, y2 = cx + cw // 2, cy + ch // 2

    cv2.namedWindow(SETUP_WINDOW, cv2.WINDOW_NORMAL)

    while True:
        canvas = disp.copy()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{title}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
        cv2.putText(canvas, "W/S/A/D move | I/K height +/- | L/J width +/- | R reset | Enter OK | Esc skip",
                    (10, canvas.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.imshow(SETUP_WINDOW, canvas)

        key = cv2.waitKey(10) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:       # Esc
            cv2.destroyWindow(SETUP_WINDOW)
            return None

        step = 10
        if key in (ord('w'), ord('W')): y1 -= step; y2 -= step
        if key in (ord('s'), ord('S')): y1 += step; y2 += step
        if key in (ord('a'), ord('A')): x1 -= step; x2 -= step
        if key in (ord('d'), ord('D')): x1 += step; x2 += step
        if key in (ord('i'), ord('I')): y1 -= step; y2 += step
        if key in (ord('k'), ord('K')): y1 += step; y2 -= step
        if key in (ord('l'), ord('L')): x1 -= step; x2 += step
        if key in (ord('j'), ord('J')): x1 += step; x2 -= step
        if key in (ord('r'), ord('R')):
            cw, ch = int(disp.shape[1] * 0.4), int(disp.shape[0] * 0.4)
            cx, cy = disp.shape[1] // 2, disp.shape[0] // 2
            x1, y1 = cx - cw // 2, cy - ch // 2
            x2, y2 = cx + cw // 2, cy + ch // 2

        # границы окна
        x1 = max(0, min(x1, disp.shape[1]-2)); x2 = max(1, min(x2, disp.shape[1]-1))
        y1 = max(0, min(y1, disp.shape[0]-2)); y2 = max(1, min(y2, disp.shape[0]-1))

    cv2.destroyWindow(SETUP_WINDOW)

    # пересчёт в оригинальные координаты
    sx, sy = int(x1 / scale), int(y1 / scale)
    ex, ey = int(x2 / scale), int(y2 / scale)
    sx, sy, ex, ey = clamp_rect(sx, sy, ex, ey, w, h)
    return [sx, sy, ex, ey]

def configure_zones(cameras: dict) -> dict:
    zones = {}
    for name, url in cameras.items():
        logging.info(f"⚙ Настройка зон для камеры '{name}'")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logging.error(f"❌ Не удалось подключиться к {name}, пропускаю.")
            zones[name] = {}
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            logging.error(f"❌ Нет кадра с {name}, пропускаю.")
            zones[name] = {}
            continue

        start_rect = select_zone_keyboard(frame, f"{name}: START")
        finish_rect = select_zone_keyboard(frame, f"{name}: FINISH")
        if start_rect and finish_rect:
            zones[name] = {"start": start_rect, "finish": finish_rect}
        else:
            zones[name] = {}
            logging.warning(f"⚠ Зоны для {name} заданы не полностью.")

    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2, ensure_ascii=False)
    logging.info("✅ Зоны сохранены в zones.json")
    cv2.destroyAllWindows()
    return zones

def preview_zones(cameras: dict, zones: dict):
    if not PREVIEW_ZONES: return
    for name, url in cameras.items():
        z = zones.get(name) or {}
        if "start" not in z or "finish" not in z:
            continue
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue
        sx1, sy1, sx2, sy2 = z["start"];  fx1, fy1, fx2, fy2 = z["finish"]
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 180, 0), 2)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
        cv2.putText(frame, "START", (sx1, max(0, sy1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,0), 2)
        cv2.putText(frame, "FINISH", (fx1, max(0, fy1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        win = f"PREVIEW_{name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, frame)
        t_end = time.time() + PREVIEW_ZONES_SECONDS
        while time.time() < t_end:
            if (cv2.waitKey(50) & 0xFF) == 27:  # Esc — досрочно
                break
        cv2.destroyWindow(win)
    cv2.destroyAllWindows()

# ===================== ДЕТЕКТОР =====================
def detect_motion_and_objects(camera_name, rtsp_url, zones_for_cam):
    """
    - детекция движения по разности;
    - при движении — YOLO;
    - центроид человека -> логика START→FINISH в DIRECTION_TIMEOUT_SEC.
    Показ кадра — через show_frame (UI-поток).
    """
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logging.error(f"❌ Не удалось подключиться к {camera_name}")
            return

        ret, frame1 = cap.read()
        ret2, frame2 = cap.read()
        if not ret or not ret2:
            logging.error(f"❌ Нет стартовых кадров у {camera_name}")
            cap.release()
            return

        last_trigger_time = 0.0
        stage = "idle"
        primed_since = 0.0
        frame_count = 0

        while True:
            # пропуски кадров ради CPU
            if frame_count % PLAYBACK_SPEED != 0:
                frame1 = frame2
                if not cap.grab(): break
                ok, frame2 = cap.retrieve()
                if not ok: break
                frame_count += 1
                continue

            # Детекция движения (на уменьшенных кадрах для экономии)
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
                # самый уверенный таргет
                best = None
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    conf = float(box.conf[0])
                    if class_name in TARGET_CLASSES and conf >= CONF_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if best is None or conf > best["conf"]:
                            best = {"class": class_name, "conf": conf, "center": (cx, cy)}

                if best:
                    cx, cy = best["center"]
                    now = time.time()
                    start_rect = zones_for_cam.get("start")
                    finish_rect = zones_for_cam.get("finish")

                    # <<< Глобальный режим: если Idle — не триггерим (но продолжаем следить за стадией)
                    active = (_get_mode() == "Primed")

                    if start_rect and finish_rect:
                        if stage == "idle":
                            if in_zone((cx, cy), start_rect):
                                stage = "primed"
                                primed_since = now
                                logging.info(f"🟢 {camera_name}: вход в START, ожидаю FINISH {DIRECTION_TIMEOUT_SEC}s")
                        elif stage == "primed":
                            if now - primed_since > DIRECTION_TIMEOUT_SEC:
                                stage = "idle"
                            elif in_zone((cx, cy), finish_rect):
                                if active and (now - last_trigger_time) >= RECOGNITION_DELAY_SEC:
                                    ts = now_ts()
                                    logging.info(f"✅ {camera_name}: START→FINISH, {best['class']} ({best['conf']:.2f}), {ts}")
                                    play_alarm()
                                    _mark_triggered()  # <<< подсветка статуса TRIGGERED
                                    # показ в UI-окне
                                    show_frame(frame2, camera_name, zones_for_cam, (cx, cy), "START→FINISH")
                                    # сохранение
                                    if SAVE_FRAMES:
                                        out = frame2.copy()
                                        sx1, sy1, sx2, sy2 = start_rect
                                        fx1, fy1, fx2, fy2 = finish_rect
                                        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 180, 0), 2)
                                        cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                                        cv2.circle(out, (cx, cy), 6, (255, 255, 0), -1)
                                        fname = f"{camera_name}_{ts.replace(':','-')}_{best['class']}.jpg"
                                        cv2.imwrite(os.path.join(date_dir(), fname), out)
                                    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                                        csv.writer(f).writerow([camera_name, ts, best["class"], f"{best['conf']:.2f}"])
                                    last_trigger_time = now
                                stage = "idle"
                    else:
                        # зон нет — обычный триггер с задержкой
                        if active and (time.time() - last_trigger_time) >= RECOGNITION_DELAY_SEC:
                            ts = now_ts()
                            logging.info(f"ℹ {camera_name}: детекция {best['class']} ({best['conf']:.2f}), {ts}")
                            play_alarm()
                            _mark_triggered()
                            show_frame(frame2, camera_name, None, (cx, cy), "DETECTED")
                            if SAVE_FRAMES:
                                fname = f"{camera_name}_{ts.replace(':','-')}_{best['class']}.jpg"
                                cv2.imwrite(os.path.join(date_dir(), fname), frame2)
                            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([camera_name, ts, best["class"], f"{best['conf']:.2f}"])
                            last_trigger_time = time.time()

            # следующий кадр
            frame1 = frame2
            if not cap.grab(): break
            ok, frame2 = cap.retrieve()
            if not ok: break
            frame_count += 1

        cap.release()

    except Exception as e:
        logging.exception(f"Ошибка при обработке {camera_name}: {e}")

# ===================== MAIN =====================
def main():
    global PLAY_ALARM, SHOW_WINDOW

    # Загрузка камер
    with open("cameras.json", "r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not cameras:
        logging.error("❌ cameras.json пустой.")
        return
    if len(cameras) > MAX_THREADS:
        logging.warning(f"⚠ Камер {len(cameras)}, рекомендуется <= {MAX_THREADS}.")

    # Настройка зон (авто)
    need_setup = True
    zones = {}
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE, "r", encoding="utf-8") as f:
                zones = json.load(f)
            need_setup = any(
                (name not in zones) or ("start" not in zones.get(name, {})) or ("finish" not in zones.get(name, {}))
                for name in cameras.keys()
            )
        except Exception:
            need_setup = True

    if need_setup:
        logging.info("⚙️ Зоны не найдены/неполные — запускаю мастер настройки…")
        zones = configure_zones(cameras)
    else:
        logging.info("✅ Найдены сохранённые зоны — мастер пропущен.")

    # Предпросмотр зон
    preview_zones(cameras, zones)

    # Вопросы (после мастера)
    try:
        PLAY_ALARM = input("Проигрывать звуковое уведомление? (y/n): ").strip().lower() == "y"
    except Exception:
        PLAY_ALARM = False
    try:
        SHOW_WINDOW = input("Показывать кадры при срабатывании (2 сек)? (y/n): ").strip().lower() == "y"
    except Exception:
        SHOW_WINDOW = True

    # UI-поток (только один, без imshow из рабочих потоков)
    ui_thread = None
    if SHOW_WINDOW:
        _view_stop.clear()
        ui_thread = threading.Thread(target=viewer_thread, daemon=True)
        ui_thread.start()

    logging.info("🔥 Горячие клавиши в окне VIEWER: Q=выход | SPACE=Idle/Primed | A=звук ON/OFF | W=показ ON/OFF")
    logging.info(f"🔍 Камер: {len(cameras)}. Запускаю в {min(len(cameras), MAX_THREADS)} поток(ах)…")

    # Запуск обработки
    try:
        with ThreadPoolExecutor(max_workers=min(len(cameras), MAX_THREADS)) as executor:
            futures = []
            for name, url in cameras.items():
                z = zones.get(name, {})
                futures.append(executor.submit(detect_motion_and_objects, name, url, z))
            # ждём все потоки
            for f in futures:
                f.result()
    finally:
        # корректно закрыть UI-поток
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
