import cv2
import time
import queue
import pygame
import threading
import numpy as np

from ultralytics import YOLO

# =========================
# AUDIO
# =========================
pygame.mixer.init()

# =========================
# MODEL
# =========================
model = YOLO("best.pt")

model.fuse()

# =========================
# STREAM
# =========================
STREAM_URL = "http://192.168.1.5"

cap = cv2.VideoCapture(
    STREAM_URL,
    cv2.CAP_FFMPEG
)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================
# VARIABLES
# =========================
fps = 0
prev = time.time()

# =========================
# ALARM
# =========================
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
# LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # =========================
    # FAST RESIZE
    # =========================
    frame = cv2.resize(
        frame,
        (640,480)
    )

    # =========================
    # YOLO
    # =========================
    results = model.predict(

        source=frame,

        imgsz=320,

        conf=0.55,

        verbose=False,

        device=0
    )

    status = "AWAKE"

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 1:

                status = "DROWSY"

                threading.Thread(
                    target=play_alarm
                ).start()

                color = (0,0,255)

            else:

                color = (0,255,0)

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
    now = time.time()

    fps = 1 / (now - prev)

    prev = now

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

    cv2.imshow(
        "ESP32 Drowsiness Detection",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()