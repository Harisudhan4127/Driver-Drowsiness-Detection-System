# =============================================================
#  DRIVER DROWSINESS DETECTION SYSTEM  v2.0
#  Detection  : MediaPipe Face Mesh  +  EAR Algorithm
#  UI         : PyQt5  —  Dark Data-Centric Dashboard
#  Author     : github.com/your-username
# =============================================================

import sys
import cv2
import time
import numpy as np
from collections import deque

import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit,
    QFrame, QGridLayout, QSpinBox, QSizePolicy,
    QSplitter
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QPainter, QPen, QColor,
    QLinearGradient, QBrush
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal


# =============================================================
#  CONSTANTS
# =============================================================
EAR_THRESHOLD     = 0.25   # Below this → eye is "closed"
EAR_CAUTION_DELTA = 0.05   # EAR < threshold + delta → caution zone
DROWSY_SECONDS    = 2.0    # Consecutive eye-closed seconds → drowsy
DANGER_SECONDS    = 4.0    # Consecutive eye-closed seconds → danger
PERCLOS_WINDOW    = 300    # Frame window for PERCLOS (~10 s @ 30 fps)
GRAPH_WINDOW      = 180    # Frames shown on the EAR graph

# MediaPipe Face Mesh — eye landmark indices
#   P1, P2, P3, P4, P5, P6  (top-left → clockwise)
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Alert levels
LEVEL_NORMAL  = 0
LEVEL_CAUTION = 1
LEVEL_DROWSY  = 2
LEVEL_DANGER  = 3

ALERT_PALETTE = {
    LEVEL_NORMAL : {"bg": "#06150a", "border": "#22c55e", "label": "ALERT",    "color": "#22c55e", "msg": "Eyes Open — Driver Alert"},
    LEVEL_CAUTION: {"bg": "#141005", "border": "#eab308", "label": "CAUTION",  "color": "#eab308", "msg": "Caution — Eyes Getting Heavy"},
    LEVEL_DROWSY : {"bg": "#130c03", "border": "#f97316", "label": "DROWSY",   "color": "#f97316", "msg": "⚠  Drowsiness Detected — Please Wake Up!"},
    LEVEL_DANGER : {"bg": "#140303", "border": "#ef4444", "label": "DANGER!",  "color": "#ef4444", "msg": "🚨  DANGER — Pull Over Immediately!"},
}


# =============================================================
#  UTILITY
# =============================================================
def euclidean(p1, p2):
    """Fast 2-D Euclidean distance (no scipy dependency)."""
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32)
                                - np.asarray(p2, dtype=np.float32)))


# =============================================================
#  DROWSINESS DETECTOR  (MediaPipe + EAR)
# =============================================================
class DrowsinessDetector:
    """
    Computes Eye Aspect Ratio (EAR) per frame via MediaPipe Face Mesh.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

    References:
      Soukupová & Čech (2016) — Real-Time Eye Blink Detection using Facial Landmarks
    """

    def __init__(self):
        self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces        = 1,
            refine_landmarks     = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )

        # Rolling buffers
        self.ear_graph    = deque(maxlen=GRAPH_WINDOW)
        self.ear_history  = deque(maxlen=PERCLOS_WINDOW)

        # State
        self.blink_count        = 0
        self.consecutive_closed = 0
        self.eye_closed_start   = None
        self.alert_level        = LEVEL_NORMAL
        self.last_ear           = 0.32

    # ----------------------------------------------------------
    def _ear(self, landmarks, indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
               for i in indices]
        A = euclidean(pts[1], pts[5])
        B = euclidean(pts[2], pts[4])
        C = euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    # ----------------------------------------------------------
    def _draw_eye(self, frame, landmarks, indices, w, h, color):
        pts = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h))
             for i in indices],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
        for p in pts:
            cv2.circle(frame, tuple(p), 2, color, -1)

    # ----------------------------------------------------------
    def process(self, frame):
        """
        Run detection on a BGR frame.
        Returns (annotated_frame, ear, face_detected).
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mp_mesh.process(rgb)
        rgb.flags.writeable = True

        ear           = self.last_ear
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark

            l_ear = self._ear(lm, LEFT_EYE_IDX,  w, h)
            r_ear = self._ear(lm, RIGHT_EYE_IDX, w, h)
            ear   = (l_ear + r_ear) / 2.0
            self.last_ear = ear

            # Eye status colour
            eye_color = (0, 255, 128) if ear >= EAR_THRESHOLD else (0, 80, 255)
            self._draw_eye(frame, lm, LEFT_EYE_IDX,  w, h, eye_color)
            self._draw_eye(frame, lm, RIGHT_EYE_IDX, w, h, eye_color)

            # Rolling data
            self.ear_graph.append(ear)
            self.ear_history.append(1 if ear < EAR_THRESHOLD else 0)

            # Blink counter
            if ear < EAR_THRESHOLD:
                self.consecutive_closed += 1
                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()
            else:
                if self.consecutive_closed >= 2:
                    self.blink_count += 1
                self.consecutive_closed = 0
                self.eye_closed_start   = None

            # Alert level
            closed_dur = (time.time() - self.eye_closed_start
                          if self.eye_closed_start else 0.0)
            perclos = self.get_perclos()

            if closed_dur >= DANGER_SECONDS or perclos > 65:
                self.alert_level = LEVEL_DANGER
            elif closed_dur >= DROWSY_SECONDS or perclos > 35:
                self.alert_level = LEVEL_DROWSY
            elif ear < EAR_THRESHOLD + EAR_CAUTION_DELTA or perclos > 15:
                self.alert_level = LEVEL_CAUTION
            else:
                self.alert_level = LEVEL_NORMAL

        else:
            self.ear_graph.append(self.last_ear)

        return frame, ear, face_detected

    # ----------------------------------------------------------
    def get_perclos(self):
        if not self.ear_history:
            return 0.0
        return (sum(self.ear_history) / len(self.ear_history)) * 100.0

    def reset(self):
        self.blink_count        = 0
        self.consecutive_closed = 0
        self.eye_closed_start   = None
        self.alert_level        = LEVEL_NORMAL
        self.ear_graph.clear()
        self.ear_history.clear()
        self.last_ear = 0.32


# =============================================================
#  CAPTURE THREAD
# =============================================================
class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error       = pyqtSignal(str)

    def __init__(self, source, is_wired=False):
        super().__init__()
        self.source   = source
        self.is_wired = is_wired
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if self.is_wired:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS,          60)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error.emit(f"Cannot open: {self.source}")
            return

        while self._running:
            ret, frame = cap.read()
            if ret and frame is not None:
                self.frame_ready.emit(frame)
            else:
                time.sleep(0.01)

        cap.release()

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)


# =============================================================
#  EAR GRAPH WIDGET  (custom QPainter)
# =============================================================
class EARGraph(QWidget):
    """Lightweight real-time EAR time-series painted with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data      : list = []
        self._threshold : float = EAR_THRESHOLD
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def push(self, data: deque):
        self._data = list(data)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        # Background
        p.fillRect(0, 0, W, H, QColor("#080d18"))

        # Horizontal grid
        p.setPen(QPen(QColor("#1a2540"), 1))
        for i in range(1, 5):
            y = int(H * i / 4)
            p.drawLine(0, y, W, y)

        # Y-axis labels
        p.setPen(QPen(QColor("#334155"), 1))
        font = QFont("Consolas", 8)
        p.setFont(font)
        for val, label in [(0.4, "0.40"), (0.25, "0.25"), (0.1, "0.10")]:
            y = int(H - (val / 0.5) * H)
            p.drawText(4, y - 2, label)

        # Threshold line
        ty = int(H - (self._threshold / 0.5) * H)
        pen = QPen(QColor("#ef4444"), 1, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(0, ty, W, ty)

        if len(self._data) < 2:
            return

        # EAR fill (gradient below line)
        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0.0, QColor(34, 211, 238, 80))
        grad.setColorAt(1.0, QColor(34, 211, 238, 0))
        step = W / max(len(self._data) - 1, 1)

        poly = [QColor(34, 211, 238)]  # placeholder; draw line manually
        pts_x = [int(i * step)                               for i in range(len(self._data))]
        pts_y = [int(H - min(max(v / 0.5, 0), 1) * H)       for v in self._data]

        pen_line = QPen(QColor(34, 211, 238), 2)
        p.setPen(pen_line)
        for i in range(1, len(pts_x)):
            p.drawLine(pts_x[i-1], pts_y[i-1], pts_x[i], pts_y[i])

        p.end()


# =============================================================
#  STAT CARD
# =============================================================
class StatCard(QFrame):
    _BASE_STYLE = """
        QFrame {{ background:{bg}; border-radius:10px; border:1px solid #1e293b; }}
    """

    def __init__(self, icon, label, value="—", unit=""):
        super().__init__()
        self.setFixedHeight(90)
        self._apply_bg("#0d1526")
        lo = QVBoxLayout(self)
        lo.setContentsMargins(10, 8, 10, 8)
        lo.setSpacing(2)

        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("color:#334155; font-size:16px; border:none;")
        lbl = QLabel(label.upper())
        lbl.setStyleSheet("color:#475569; font-size:10px; font-weight:bold; letter-spacing:1px; border:none;")
        top.addWidget(icon_lbl)
        top.addWidget(lbl)
        top.addStretch()

        self.val_lbl = QLabel(str(value))
        self.val_lbl.setFont(QFont("Consolas", 20, QFont.Bold))
        self.val_lbl.setStyleSheet("color:#e2e8f0; border:none;")

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setStyleSheet("color:#334155; font-size:10px; border:none;")

        lo.addLayout(top)
        lo.addWidget(self.val_lbl)
        lo.addWidget(self.unit_lbl)

    def _apply_bg(self, color):
        self.setStyleSheet(self._BASE_STYLE.format(bg=color))

    def set_value(self, v: str, color="#e2e8f0"):
        self.val_lbl.setText(v)
        self.val_lbl.setStyleSheet(f"color:{color}; border:none;")


# =============================================================
#  MAIN APPLICATION WINDOW
# =============================================================
class DrowsinessApp(QWidget):

    def __init__(self):
        super().__init__()
        self.detector       = DrowsinessDetector()
        self.capture_thread : CaptureThread | None = None
        self._latest_frame  : np.ndarray | None    = None
        self._prev_time     = time.time()
        self._fps           = 0
        self._setup_ui()

    # ----------------------------------------------------------
    #  UI CONSTRUCTION
    # ----------------------------------------------------------
    def _setup_ui(self):
        self.setWindowTitle("Driver Drowsiness Detection System  v2.0")
        self.showMaximized()
        self.setStyleSheet("""
            QWidget {
                background-color : #060c18;
                color            : #cbd5e1;
                font-family      : 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel  { border: none; }
            QPushButton {
                background    : #1a3a6b;
                color         : #e2e8f0;
                border        : 1px solid #2563eb;
                border-radius : 7px;
                padding       : 8px 18px;
                font-size     : 12px;
                font-weight   : bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover   { background: #2563eb; border-color:#3b82f6; }
            QPushButton:pressed { background: #1d4ed8; }
            QPushButton:disabled{ background: #0f172a; color:#334155; border-color:#1e293b; }
            QPushButton#stop {
                background  : #3b0a0a;
                border-color: #dc2626;
            }
            QPushButton#stop:hover { background:#7f1d1d; }
            QPushButton#reset {
                background  : #0f2a1a;
                border-color: #166534;
            }
            QPushButton#reset:hover { background:#14532d; }
            QComboBox, QLineEdit, QSpinBox {
                background    : #0d1526;
                border        : 1px solid #1e3a5f;
                border-radius : 7px;
                padding       : 7px 10px;
                color         : #cbd5e1;
                font-size     : 13px;
            }
            QComboBox::drop-down  { border:none; width:24px; }
            QComboBox::down-arrow { width:10px; }
            QComboBox QAbstractItemView {
                background          : #0d1526;
                color               : #cbd5e1;
                selection-background: #1d4ed8;
            }
            QSpinBox::up-button, QSpinBox::down-button { width:18px; }
        """)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(14, 10, 14, 10)

        root.addWidget(self._build_header())
        root.addWidget(self._build_control_bar())

        # ── Content ──────────────────────────────────────────
        content_lo = QHBoxLayout()
        content_lo.setSpacing(10)

        # Video pane
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame { background:#000308; border-radius:12px; border:1px solid #0f2040; }
        """)
        vf_lo = QVBoxLayout(video_frame)
        vf_lo.setContentsMargins(0, 0, 0, 0)
        self._video_lbl = QLabel("NO SIGNAL")
        self._video_lbl.setAlignment(Qt.AlignCenter)
        self._video_lbl.setStyleSheet(
            "color:#1e3a5f; font-size:22px; font-weight:bold; "
            "letter-spacing:4px; background:transparent; border:none;"
        )
        self._video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vf_lo.addWidget(self._video_lbl)
        content_lo.addWidget(video_frame, stretch=7)

        # Right panel
        content_lo.addLayout(self._build_right_panel(), stretch=3)
        root.addLayout(content_lo, stretch=1)

        root.addWidget(self._build_footer())

        # Timer: update UI from latest frame ~30 fps
        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._process_latest_frame)

    # ──────────────────────────────────────────────────────────
    def _build_header(self):
        f = QFrame()
        f.setFixedHeight(52)
        f.setStyleSheet("""
            QFrame { background:#07101f; border-radius:10px; border:1px solid #0f2040; }
        """)
        lo = QHBoxLayout(f)
        lo.setContentsMargins(18, 0, 18, 0)

        pulse = QLabel("◉")
        pulse.setStyleSheet("color:#22c55e; font-size:18px;")
        title = QLabel("DRIVER DROWSINESS DETECTION SYSTEM")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setStyleSheet("color:#e2e8f0; letter-spacing:3px;")
        sub = QLabel("MediaPipe Face Mesh  ·  EAR Algorithm  ·  Real-time Analysis")
        sub.setStyleSheet("color:#334155; font-size:11px;")

        lo.addWidget(pulse)
        lo.addSpacing(10)
        lo.addWidget(title)
        lo.addSpacing(20)
        lo.addWidget(sub)
        lo.addStretch()

        self._conn_dot = QLabel("●  DISCONNECTED")
        self._conn_dot.setStyleSheet("color:#dc2626; font-size:11px; font-weight:bold; letter-spacing:1px;")
        lo.addWidget(self._conn_dot)
        return f

    # ──────────────────────────────────────────────────────────
    def _build_control_bar(self):
        f = QFrame()
        f.setFixedHeight(58)
        f.setStyleSheet("""
            QFrame { background:#07101f; border-radius:10px; border:1px solid #0f2040; }
        """)
        lo = QHBoxLayout(f)
        lo.setContentsMargins(18, 0, 18, 0)
        lo.setSpacing(10)

        def lbl(text):
            l = QLabel(text)
            l.setStyleSheet("color:#334155; font-size:10px; font-weight:bold; letter-spacing:1px;")
            return l

        # Mode selector
        lo.addWidget(lbl("MODE"))
        self._mode_box = QComboBox()
        self._mode_box.addItems(["Wireless  (ESP32-CAM)", "Wired  (USB Camera)"])
        self._mode_box.setFixedWidth(190)
        self._mode_box.currentIndexChanged.connect(self._on_mode_change)
        lo.addWidget(self._mode_box)
        lo.addSpacing(6)

        # IP / Camera Index
        lo.addWidget(lbl("IP / INDEX"))
        self._ip_edit = QLineEdit("192.168.1.5")
        self._ip_edit.setFixedWidth(145)
        lo.addWidget(self._ip_edit)

        # Port
        lo.addWidget(lbl("PORT"))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(80)
        self._port_spin.setFixedWidth(80)
        lo.addWidget(self._port_spin)

        # Path
        lo.addWidget(lbl("PATH"))
        self._path_edit = QLineEdit("/")
        self._path_edit.setFixedWidth(60)
        lo.addWidget(self._path_edit)

        lo.addSpacing(12)

        # Buttons
        self._btn_connect = QPushButton("▶  CONNECT")
        self._btn_connect.clicked.connect(self._connect)
        self._btn_connect.setFixedHeight(36)

        self._btn_stop = QPushButton("■  STOP")
        self._btn_stop.setObjectName("stop")
        self._btn_stop.clicked.connect(self._disconnect)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setFixedHeight(36)

        self._btn_reset = QPushButton("↺  RESET")
        self._btn_reset.setObjectName("reset")
        self._btn_reset.clicked.connect(self._reset)
        self._btn_reset.setFixedHeight(36)

        lo.addWidget(self._btn_connect)
        lo.addWidget(self._btn_stop)
        lo.addWidget(self._btn_reset)
        lo.addStretch()

        return f

    # ──────────────────────────────────────────────────────────
    def _build_right_panel(self):
        lo = QVBoxLayout()
        lo.setSpacing(8)

        # Alert banner
        self._alert_frame = QFrame()
        self._alert_frame.setFixedHeight(88)
        alert_lo = QVBoxLayout(self._alert_frame)
        alert_lo.setContentsMargins(12, 8, 12, 8)
        alert_lo.setSpacing(4)
        self._alert_lbl = QLabel("ALERT")
        self._alert_lbl.setFont(QFont("Consolas", 26, QFont.Bold))
        self._alert_lbl.setAlignment(Qt.AlignCenter)
        self._alert_msg = QLabel("Eyes Open — Driver Alert")
        self._alert_msg.setAlignment(Qt.AlignCenter)
        self._alert_msg.setStyleSheet("color:#64748b; font-size:11px;")
        alert_lo.addWidget(self._alert_lbl)
        alert_lo.addWidget(self._alert_msg)
        self._apply_alert(LEVEL_NORMAL)
        lo.addWidget(self._alert_frame)

        # Stat cards
        grid = QGridLayout()
        grid.setSpacing(7)
        self._card_ear     = StatCard("◎", "EAR",         "—",  "ratio")
        self._card_perclos = StatCard("⏱", "PERCLOS",     "—",  "%")
        self._card_blinks  = StatCard("👁", "BLINKS",      "0",  "count")
        self._card_fps     = StatCard("⚡", "FPS",         "—",  "frames/s")
        grid.addWidget(self._card_ear,     0, 0)
        grid.addWidget(self._card_perclos, 0, 1)
        grid.addWidget(self._card_blinks,  1, 0)
        grid.addWidget(self._card_fps,     1, 1)
        lo.addLayout(grid)

        # Graph
        g_header = QHBoxLayout()
        g_lbl = QLabel("EAR TIME SERIES")
        g_lbl.setStyleSheet("color:#334155; font-size:10px; font-weight:bold; letter-spacing:1px;")
        g_thresh = QLabel("── 0.25 threshold")
        g_thresh.setStyleSheet("color:#ef4444; font-size:10px;")
        g_header.addWidget(g_lbl)
        g_header.addStretch()
        g_header.addWidget(g_thresh)
        lo.addLayout(g_header)

        self._ear_graph = EARGraph()
        lo.addWidget(self._ear_graph)

        # Face status
        self._face_status = QLabel("● FACE NOT DETECTED")
        self._face_status.setAlignment(Qt.AlignCenter)
        self._face_status.setStyleSheet(
            "color:#dc2626; font-size:11px; font-weight:bold; "
            "background:#0d1526; border-radius:6px; border:1px solid #1e293b; "
            "padding:5px;"
        )
        lo.addWidget(self._face_status)
        lo.addStretch()
        return lo

    # ──────────────────────────────────────────────────────────
    def _build_footer(self):
        f = QFrame()
        f.setFixedHeight(32)
        f.setStyleSheet("""
            QFrame { background:#07101f; border-radius:8px; border:1px solid #0f2040; }
        """)
        lo = QHBoxLayout(f)
        lo.setContentsMargins(14, 0, 14, 0)
        self._status_lbl = QLabel("System ready — connect a camera to begin.")
        self._status_lbl.setStyleSheet("color:#334155; font-size:11px;")
        model_lbl = QLabel("Model: MediaPipe Face Mesh  |  Algorithm: EAR  |  v2.0")
        model_lbl.setStyleSheet("color:#1e3a5f; font-size:11px;")
        lo.addWidget(self._status_lbl)
        lo.addStretch()
        lo.addWidget(model_lbl)
        return f

    # ----------------------------------------------------------
    #  CAMERA CONTROL
    # ----------------------------------------------------------
    def _on_mode_change(self, idx):
        wireless = idx == 0
        self._ip_edit.setText("192.168.1.5" if wireless else "0")
        self._ip_edit.setPlaceholderText("192.168.1.5" if wireless else "0  (camera index)")
        self._port_spin.setEnabled(wireless)
        self._path_edit.setEnabled(wireless)

    def _connect(self):
        wireless = self._mode_box.currentIndex() == 0

        if wireless:
            ip   = self._ip_edit.text().strip()
            port = self._port_spin.value()
            path = self._path_edit.text().strip()
            if not ip.startswith("http"):
                ip = f"http://{ip}"
            source = f"{ip}:{port}{path}"
        else:
            raw = self._ip_edit.text().strip()
            try:
                source = int(raw)
            except ValueError:
                source = 0

        self.detector.reset()
        self._latest_frame = None

        self.capture_thread = CaptureThread(source, is_wired=not wireless)
        self.capture_thread.frame_ready.connect(self._on_new_frame)
        self.capture_thread.error.connect(self._on_capture_error)
        self.capture_thread.start()

        self._ui_timer.start(30)   # 30 ms ≈ 33 fps UI refresh

        self._btn_connect.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._conn_dot.setText("●  CONNECTED")
        self._conn_dot.setStyleSheet("color:#22c55e; font-size:11px; font-weight:bold; letter-spacing:1px;")
        self._status_lbl.setText(f"Connected → {source}")

    def _disconnect(self):
        self._ui_timer.stop()
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        self._latest_frame = None
        self._video_lbl.setPixmap(QPixmap())
        self._video_lbl.setText("NO SIGNAL")
        self._btn_connect.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._conn_dot.setText("●  DISCONNECTED")
        self._conn_dot.setStyleSheet("color:#dc2626; font-size:11px; font-weight:bold; letter-spacing:1px;")
        self._status_lbl.setText("Disconnected.")

    def _reset(self):
        self.detector.reset()
        self._status_lbl.setText("Statistics reset.")

    # ----------------------------------------------------------
    #  FRAME PIPELINE
    # ----------------------------------------------------------
    def _on_new_frame(self, frame: np.ndarray):
        """Slot — just store the latest frame (called from capture thread)."""
        self._latest_frame = frame

    def _on_capture_error(self, msg: str):
        self._status_lbl.setText(f"Error: {msg}")
        self._disconnect()

    def _process_latest_frame(self):
        """Timer slot — runs on main thread, performs detection and UI update."""
        if self._latest_frame is None:
            return

        frame = self._latest_frame.copy()

        # FPS
        now = time.time()
        elapsed = max(now - self._prev_time, 1e-6)
        self._fps = int(1.0 / elapsed)
        self._prev_time = now

        # ── Detection ─────────────────────────────────────────
        frame, ear, face_detected = self.detector.process(frame)
        level   = self.detector.alert_level
        perclos = self.detector.get_perclos()

        # ── Overlay ───────────────────────────────────────────
        overlay_color = {
            LEVEL_NORMAL : (0, 220, 100),
            LEVEL_CAUTION: (0, 210, 220),
            LEVEL_DROWSY : (0, 140, 255),
            LEVEL_DANGER : (0, 60,  255),
        }[level]

        cv2.putText(frame, f"EAR: {ear:.3f}",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, overlay_color, 2)
        cv2.putText(frame, f"FPS: {self._fps}",
                    (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%",
                    (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

        alert_txt = ALERT_PALETTE[level]["label"]
        if level >= LEVEL_DROWSY:
            cv2.putText(frame, f"!! {alert_txt} !!",
                        (14, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255) if level == LEVEL_DANGER else (0, 100, 255),
                        3)

        # ── Display ───────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img    = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            self._video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._video_lbl.setPixmap(pixmap)

        # ── Stats cards ───────────────────────────────────────
        level_colors = {
            LEVEL_NORMAL : "#22c55e",
            LEVEL_CAUTION: "#eab308",
            LEVEL_DROWSY : "#f97316",
            LEVEL_DANGER : "#ef4444",
        }
        c = level_colors[level]
        self._card_ear.set_value(f"{ear:.3f}", c)
        self._card_perclos.set_value(f"{perclos:.1f}", c)
        self._card_blinks.set_value(str(self.detector.blink_count))
        self._card_fps.set_value(str(self._fps))

        # ── Alert banner ──────────────────────────────────────
        self._apply_alert(level)

        # ── Face status ───────────────────────────────────────
        if face_detected:
            self._face_status.setText("●  FACE DETECTED")
            self._face_status.setStyleSheet(
                "color:#22c55e; font-size:11px; font-weight:bold; "
                "background:#06150a; border-radius:6px; border:1px solid #166534; padding:5px;"
            )
        else:
            self._face_status.setText("●  FACE NOT DETECTED")
            self._face_status.setStyleSheet(
                "color:#dc2626; font-size:11px; font-weight:bold; "
                "background:#14040a; border-radius:6px; border:1px solid #7f1d1d; padding:5px;"
            )

        # ── Graph ─────────────────────────────────────────────
        self._ear_graph.push(self.detector.ear_graph)

    # ----------------------------------------------------------
    def _apply_alert(self, level: int):
        p = ALERT_PALETTE[level]
        self._alert_frame.setStyleSheet(f"""
            QFrame {{
                background   : {p['bg']};
                border-radius: 12px;
                border       : 2px solid {p['border']};
            }}
        """)
        self._alert_lbl.setText(p["label"])
        self._alert_lbl.setStyleSheet(
            f"color:{p['color']}; font-size:26px; font-weight:bold; border:none;"
        )
        self._alert_msg.setText(p["msg"])

    def closeEvent(self, event):
        self._disconnect()
        event.accept()


# =============================================================
#  ENTRY POINT
# =============================================================
if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DrowsinessApp()
    win.show()
    sys.exit(app.exec_())
