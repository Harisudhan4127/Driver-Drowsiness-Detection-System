import cv2
import time
import pygame
import threading

from ultralytics import YOLO

# =========================
# OPENCV OPTIMIZATION
# =========================
cv2.setUseOptimized(True)

cv2.setNumThreads(8)

# =========================
# AUDIO
# =========================
pygame.mixer.init()

alarm_playing = False

def play_alarm():

    global alarm_playing

    if not alarm_playing:

        alarm_playing = True

        pygame.mixer.music.load(
            "alarm.mp3"
        )

        pygame.mixer.music.play()

# =========================
# LOAD ONNX MODEL
# =========================
model = YOLO("yolov8n.onnx")

# =========================
# ESP32 STREAM
# =========================
STREAM_URL = "http://192.168.1.5"

cap = cv2.VideoCapture(
    STREAM_URL,
    cv2.CAP_FFMPEG
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

# =========================
# FPS
# =========================
prev_time = time.time()

# =========================
# LOOP
# =========================
while True:

    success, frame = cap.read()

    if not success:
        continue

    # =========================
    # FAST RESIZE
    # =========================
    frame = cv2.resize(
        frame,
        (640,480)
    )

    # =========================
    # YOLO ONNX
    # =========================
    results = model.predict(

        source=frame,

        imgsz=320,

        conf=0.5,

        verbose=False
    )

    status = "AWAKE"

    # =========================
    # DETECTIONS
    # =========================
    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            if cls == 1:

                color = (0,0,255)

                status = "DROWSY"

                threading.Thread(
                    target=play_alarm
                ).start()

            else:

                color = (0,255,0)

            # =========================
            # DRAW
            # =========================
            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{status} {conf:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # =========================
    # FPS
    # =========================
    current_time = time.time()

    fps = 1 / (
        current_time - prev_time
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        status,
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow(
        "ESP32-CAM Drowsiness Detection",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

# =========================
# RELEASE
# =========================
cap.release()

cv2.destroyAllWindows()