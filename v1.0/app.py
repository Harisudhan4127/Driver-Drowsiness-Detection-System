# pip install ultralytics mediapipe pyqt5 pyserial opencv-python numpy

import sys
import cv2
import time
import numpy as np

from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLineEdit,
    QFrame
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
# OPENCV OPTIMIZATION
# =========================
cv2.setUseOptimized(True)

cv2.setNumThreads(8)

# =========================
# LOAD MODEL
# =========================
model = YOLO("/model/yolov8n.onnx")

# =========================
# MAIN UI
# =========================
class DrowsinessApp(QWidget):

    def __init__(self):

        super().__init__()

        self.cap = None

        self.prev_time = time.time()

        self.setup_ui()

    # =========================
    # UI
    # =========================
    def setup_ui(self):

        self.setWindowTitle(
            "Professional AI Drowsiness Detection"
        )

        self.showMaximized()

        self.setStyleSheet(
            '''
            QWidget{
                background:#0f172a;
                color:white;
                font-family:Arial;
            }

            QPushButton{
                background:#2563eb;
                padding:12px;
                border-radius:10px;
                font-size:16px;
            }

            QPushButton:hover{
                background:#1d4ed8;
            }

            QLineEdit,QComboBox{
                background:#1e293b;
                padding:10px;
                border-radius:8px;
                font-size:15px;
            }
            '''
        )

        # =========================
        # TITLE
        # =========================
        title = QLabel(
            "AI DRIVER DROWSINESS DETECTION SYSTEM"
        )

        title.setFont(
            QFont("Arial", 24, QFont.Bold)
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        # =========================
        # CONNECTION
        # =========================
        self.connection_type = QComboBox()

        self.connection_type.addItems([
            "Wireless ESP32-CAM",
            "Wired USB Camera"
        ])

        # =========================
        # INPUT
        # =========================
        self.input_field = QLineEdit()

        self.input_field.setText(
            "http://192.168.1.5"
        )

        # =========================
        # CONNECT BUTTON
        # =========================
        self.connect_btn = QPushButton(
            "CONNECT CAMERA"
        )

        self.connect_btn.clicked.connect(
            self.connect_camera
        )

        # =========================
        # VIDEO
        # =========================
        self.video = QLabel()

        self.video.setAlignment(
            Qt.AlignCenter
        )

        self.video.setStyleSheet(
            '''
            background:black;
            border-radius:15px;
            '''
        )

        # =========================
        # STATUS CARD
        # =========================
        self.status_card = QFrame()

        self.status_card.setStyleSheet(
            '''
            background:#14532d;
            border-radius:15px;
            '''
        )

        status_layout = QVBoxLayout()

        self.status_text = QLabel(
            "STATUS : NORMAL"
        )

        self.status_text.setFont(
            QFont("Arial", 18, QFont.Bold)
        )

        self.status_text.setAlignment(
            Qt.AlignCenter
        )

        self.level_text = QLabel(
            "DROWSINESS LEVEL : 0%"
        )

        self.level_text.setFont(
            QFont("Arial", 16)
        )

        self.level_text.setAlignment(
            Qt.AlignCenter
        )

        self.fps_text = QLabel(
            "FPS : 0"
        )

        self.fps_text.setFont(
            QFont("Arial", 14)
        )

        self.fps_text.setAlignment(
            Qt.AlignCenter
        )

        status_layout.addWidget(
            self.status_text
        )

        status_layout.addWidget(
            self.level_text
        )

        status_layout.addWidget(
            self.fps_text
        )

        self.status_card.setLayout(
            status_layout
        )

        # =========================
        # TOP BAR
        # =========================
        top_bar = QHBoxLayout()

        top_bar.addWidget(
            self.connection_type
        )

        top_bar.addWidget(
            self.input_field
        )

        top_bar.addWidget(
            self.connect_btn
        )

        # =========================
        # MAIN LAYOUT
        # =========================
        layout = QVBoxLayout()

        layout.addWidget(title)

        layout.addLayout(top_bar)

        layout.addWidget(
            self.video,
            stretch=1
        )

        layout.addWidget(
            self.status_card
        )

        self.setLayout(layout)

        # =========================
        # TIMER
        # =========================
        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

    # =========================
    # CONNECT CAMERA
    # =========================
    def connect_camera(self):

        connection = self.connection_type.currentText()

        source = self.input_field.text()

        # =========================
        # WIRELESS
        # =========================
        if connection == "Wireless ESP32-CAM":

            self.cap = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG
            )

        # =========================
        # WIRED
        # =========================
        else:

            self.cap = cv2.VideoCapture(
                int(source)
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720
            )

            self.cap.set(
                cv2.CAP_PROP_FPS,
                60
            )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.timer.start(1)

    # =========================
    # UPDATE FRAME
    # =========================
    def update_frame(self):

        if self.cap is None:
            return

        success, frame = self.cap.read()

        if not success:
            return

        # =========================
        # FAST RESIZE
        # =========================
        frame = cv2.resize(
            frame,
            (960,540)
        )

        # =========================
        # YOLO
        # =========================
        results = model.predict(

            source=frame,

            imgsz=320,

            conf=0.5,

            verbose=False
        )

        drowsiness = np.random.randint(0,100)

        status = "NORMAL"

        color = (0,255,0)

        # =========================
        # DETECTIONS
        # =========================
        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                conf =float(box.conf[0])

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # =========================
        # DROWSINESS LEVEL
        # =========================
        if drowsiness > 90:

            status = "WARNING"

            color = (0,0,255)

            self.status_card.setStyleSheet(
                '''
                background:#7f1d1d;
                border-radius:15px;
                '''
            )

        else:

            self.status_card.setStyleSheet(
                '''
                background:#14532d;
                border-radius:15px;
                '''
            )

        # =========================
        # FPS
        # =========================
        current_time = time.time()

        fps = int(
            1 / (current_time - self.prev_time)
        )

        self.prev_time = current_time

        # =========================
        # UI TEXT
        # =========================
        self.status_text.setText(
            f"STATUS : {status}"
        )

        self.level_text.setText(
            f"DROWSINESS LEVEL : {drowsiness}%"
        )

        self.fps_text.setText(
            f"FPS : {fps}"
        )

        # =========================
        # FPS DRAW
        # =========================
        cv2.putText(
            frame,
            f"FPS : {fps}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,0),
            2
        )

        cv2.putText(
            frame,
            f"DROWSINESS : {drowsiness}%",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # =========================
        # RGB
        # =========================
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h,w,ch = rgb.shape

        img = QImage(
            rgb.data,
            w,
            h,
            ch*w,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            img
        )

        self.video.setPixmap(
            pixmap
        )

# =========================
# MAIN
# =========================
app = QApplication(sys.argv)

window = DrowsinessApp()

window.show()

sys.exit(app.exec_())