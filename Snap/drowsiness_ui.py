import sys
import cv2
import pygame
import threading

from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout
)

from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QFont
)

from PyQt5.QtCore import (
    Qt,
    QTimer
)

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

def stop_alarm():

    global alarm_playing

    pygame.mixer.music.stop()

    alarm_playing = False

# =========================
# YOLO MODEL
# =========================
model = YOLO("yolov8s.pt")

model.fuse()

# =========================
# STREAM URL
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
# UI
# =========================
class App(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Real-Time Drowsiness Detection"
        )

        self.showMaximized()

        self.setStyleSheet(
            "background:#0f172a;"
        )

        self.video = QLabel()

        self.video.setAlignment(
            Qt.AlignCenter
        )

        self.status = QLabel(
            "STATUS : AWAKE"
        )

        self.status.setAlignment(
            Qt.AlignCenter
        )

        self.status.setFont(
            QFont("Arial", 20)
        )

        self.status.setStyleSheet(
            """
            color:white;
            background:#14532d;
            padding:15px;
            border-radius:10px;
            """
        )

        layout = QVBoxLayout()

        layout.addWidget(self.video)

        layout.addWidget(self.status)

        self.setLayout(layout)

        self.closed_frames = 0

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

        self.timer.start(1)

    # =========================
    # UPDATE FRAME
    # =========================
    def update_frame(self):

        success, frame = cap.read()

        if not success:
            return

        frame = cv2.resize(
            frame,
            (1280,720)
        )

        results = model(
            frame,
            imgsz=640,
            conf=0.5,
            verbose=False
        )

        drowsy = False

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                face_height = y2 - y1

                # =========================
                # SIMPLE DROWSINESS
                # =========================
                if face_height < 150:

                    self.closed_frames += 1

                else:

                    self.closed_frames = 0

        if self.closed_frames > 15:

            drowsy = True

            self.status.setText(
                "STATUS : DROWSY ALERT"
            )

            self.status.setStyleSheet(
                """
                color:white;
                background:#7f1d1d;
                padding:15px;
                border-radius:10px;
                """
            )

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (350,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,0,255),
                4
            )

            threading.Thread(
                target=play_alarm
            ).start()

        else:

            self.status.setText(
                "STATUS : AWAKE"
            )

            self.status.setStyleSheet(
                """
                color:white;
                background:#14532d;
                padding:15px;
                border-radius:10px;
                """
            )

            stop_alarm()

        # =========================
        # RGB
        # =========================
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        img = QImage(
            rgb.data,
            w,
            h,
            ch*w,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(img)

        self.video.setPixmap(
            pixmap
        )

# =========================
# MAIN
# =========================
app = QApplication(sys.argv)

window = App()

window.show()

sys.exit(app.exec_())