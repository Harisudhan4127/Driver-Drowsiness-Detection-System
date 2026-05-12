# =============================================================
#  DRIVER DROWSINESS DETECTION SYSTEM  v3.0
#  Detection  : MediaPipe Face Mesh  +  EAR Algorithm
#  UI         : PyQt5  —  Fluent Design System  (Colorful Theme)
#  New        : ESP32-CAM Serial Port (COM port) support
#  Author     : github.com/your-username
# =============================================================

import sys
import cv2
import time
import serial
import serial.tools.list_ports
import numpy as np
from collections import deque

import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit,
    QFrame, QGridLayout, QSpinBox, QSizePolicy,
    QSplitter, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QPainter, QPen, QColor,
    QLinearGradient, QBrush, QGradient, QPainterPath,
    QPalette
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QRect


# =============================================================
#  FLUENT DESIGN COLOUR PALETTE
# =============================================================
F = {
    # Surfaces
    "surface_bg":   "#F3F2F1",
    "surface_card": "#FFFFFF",
    "surface_deep": "#EDEBE9",
    "surface_bar":  "#FFFFFF",

    # Brand / Accent
    "accent":       "#0078D7",
    "accent_hover": "#005A9E",
    "accent_press": "#004275",
    "accent_light": "#C7E0F4",

    # Interactive teal
    "teal":         "#00B7C3",
    "teal_light":   "#BFF5F8",

    # Text
    "text_primary": "#201F1E",
    "text_second":  "#605E5C",
    "text_disabled":"#A19F9D",
    "text_on_acc":  "#FFFFFF",

    # Status
    "success":      "#107C10",
    "success_bg":   "#DFF6DD",
    "warning":      "#FFB900",
    "warning_bg":   "#FFF4CE",
    "danger":       "#E81123",
    "danger_bg":    "#FDE7E9",
    "alert_orange": "#F7630C",
    "alert_bg":     "#FFDFD3",

    # Data viz palette
    "dv_purple":    "#8764B8",
    "dv_red":       "#EF6950",
    "dv_blue":      "#0078D7",
    "dv_teal":      "#00B7C3",
    "dv_green":     "#107C10",
    "dv_orange":    "#F7630C",

    # Border / Divider
    "border":       "#EDEBE9",
    "border_focus": "#0078D7",
}

# =============================================================
#  CONSTANTS
# =============================================================
EAR_THRESHOLD     = 0.25
EAR_CAUTION_DELTA = 0.05
DROWSY_SECONDS    = 2.0
DANGER_SECONDS    = 4.0
PERCLOS_WINDOW    = 300
GRAPH_WINDOW      = 180

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

LEVEL_NORMAL  = 0
LEVEL_CAUTION = 1
LEVEL_DROWSY  = 2
LEVEL_DANGER  = 3

ALERT_PALETTE = {
    LEVEL_NORMAL : {
        "bg": F["success_bg"], "border": F["success"],
        "label": "ALERT",    "color": F["success"],
        "msg": "Eyes Open — Driver Alert",
        "icon": "✔"
    },
    LEVEL_CAUTION: {
        "bg": F["warning_bg"], "border": F["warning"],
        "label": "CAUTION",  "color": "#8A5C00",
        "msg": "Caution — Eyes Getting Heavy",
        "icon": "⚠"
    },
    LEVEL_DROWSY : {
        "bg": F["alert_bg"], "border": F["alert_orange"],
        "label": "DROWSY",   "color": F["alert_orange"],
        "msg": "Drowsiness Detected — Please Wake Up!",
        "icon": "😴"
    },
    LEVEL_DANGER : {
        "bg": F["danger_bg"], "border": F["danger"],
        "label": "DANGER!",  "color": F["danger"],
        "msg": "🚨  DANGER — Pull Over Immediately!",
        "icon": "🚨"
    },
}


# =============================================================
#  UTILITY
# =============================================================
def euclidean(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32)
                                - np.asarray(p2, dtype=np.float32)))

def get_serial_ports():
    """Return list of available COM/serial ports with descriptions."""
    ports = serial.tools.list_ports.comports()
    result = []
    for p in sorted(ports):
        desc = p.description or p.name
        result.append((p.device, desc))
    return result


# =============================================================
#  DROWSINESS DETECTOR
# =============================================================
class DrowsinessDetector:
    def __init__(self):
        self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.ear_graph    = deque(maxlen=GRAPH_WINDOW)
        self.ear_history  = deque(maxlen=PERCLOS_WINDOW)
        self.blink_count        = 0
        self.consecutive_closed = 0
        self.eye_closed_start   = None
        self.alert_level        = LEVEL_NORMAL
        self.last_ear           = 0.32

    def _ear(self, landmarks, indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        A = euclidean(pts[1], pts[5])
        B = euclidean(pts[2], pts[4])
        C = euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    def _draw_eye(self, frame, landmarks, indices, w, h, color):
        pts = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
        for p in pts:
            cv2.circle(frame, tuple(p), 2, color, -1)

    def process(self, frame):
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

            eye_color = (16, 124, 16) if ear >= EAR_THRESHOLD else (232, 17, 35)
            self._draw_eye(frame, lm, LEFT_EYE_IDX,  w, h, eye_color)
            self._draw_eye(frame, lm, RIGHT_EYE_IDX, w, h, eye_color)

            self.ear_graph.append(ear)
            self.ear_history.append(1 if ear < EAR_THRESHOLD else 0)

            if ear < EAR_THRESHOLD:
                self.consecutive_closed += 1
                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()
            else:
                if self.consecutive_closed >= 2:
                    self.blink_count += 1
                self.consecutive_closed = 0
                self.eye_closed_start   = None

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
#  CAPTURE THREAD  (supports wireless HTTP, wired USB, serial COM)
# =============================================================
class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error       = pyqtSignal(str)

    MODE_WIRELESS = 0   # HTTP MJPEG stream
    MODE_WIRED    = 1   # OpenCV camera index
    MODE_SERIAL   = 2   # ESP32-CAM over serial COM port

    def __init__(self, source, mode=MODE_WIRELESS):
        super().__init__()
        self.source   = source
        self.mode     = mode
        self._running = True

    def run(self):
        if self.mode == self.MODE_SERIAL:
            self._run_serial()
        else:
            self._run_cv()

    # ── OpenCV path (wireless or wired) ───────────────────────
    def _run_cv(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if self.mode == self.MODE_WIRED:
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

    # ── Serial (COM port) path — reads MJPEG from ESP32-CAM ───
    def _run_serial(self):
        """
        Reads JPEG frames sent over UART by ESP32-CAM.
        The firmware must push raw JPEG bytes delimited by
        the standard SOI (FF D8) / EOI (FF D9) JPEG markers.
        """
        try:
            ser = serial.Serial(self.source, baudrate=921600, timeout=2)
        except serial.SerialException as e:
            self.error.emit(f"Serial error: {e}")
            return

        buf = b""
        while self._running:
            try:
                chunk = ser.read(4096)
                if not chunk:
                    continue
                buf += chunk

                while True:
                    start = buf.find(b'\xff\xd8')
                    end   = buf.find(b'\xff\xd9')
                    if start == -1 or end == -1 or end < start:
                        break
                    jpeg_bytes = buf[start:end + 2]
                    buf = buf[end + 2:]

                    img_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame_ready.emit(frame)

            except serial.SerialException as e:
                self.error.emit(f"Serial read error: {e}")
                break
            except Exception:
                time.sleep(0.01)

        try:
            ser.close()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)


# =============================================================
#  FLUENT EAR GRAPH WIDGET
# =============================================================
class EARGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data      : list = []
        self._threshold : float = EAR_THRESHOLD
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"background:{F['surface_card']}; border-radius:8px;")

    def push(self, data: deque):
        self._data = list(data)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        # White background with subtle border
        p.fillRect(0, 0, W, H, QColor(F["surface_card"]))
        p.setPen(QPen(QColor(F["border"]), 1))
        p.drawRect(0, 0, W - 1, H - 1)

        # Horizontal grid lines (light gray)
        p.setPen(QPen(QColor("#EDEBE9"), 1))
        for i in range(1, 5):
            y = int(H * i / 4)
            p.drawLine(0, y, W, y)

        # Y-axis labels
        p.setPen(QPen(QColor(F["text_disabled"]), 1))
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        for val, label in [(0.40, "0.40"), (0.25, "0.25"), (0.10, "0.10")]:
            y = int(H - (val / 0.5) * H)
            p.drawText(4, y - 2, label)

        # Threshold dashed line (danger red)
        ty = int(H - (self._threshold / 0.5) * H)
        pen = QPen(QColor(F["danger"]), 1, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(0, ty, W, ty)

        if len(self._data) < 2:
            p.end()
            return

        step  = W / max(len(self._data) - 1, 1)
        pts_x = [int(i * step)                          for i in range(len(self._data))]
        pts_y = [int(H - min(max(v / 0.5, 0), 1) * H)  for v in self._data]

        # Fill area under curve — teal gradient
        path = QPainterPath()
        path.moveTo(pts_x[0], H)
        for i in range(len(pts_x)):
            path.lineTo(pts_x[i], pts_y[i])
        path.lineTo(pts_x[-1], H)
        path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0.0, QColor(0, 183, 195, 80))
        grad.setColorAt(1.0, QColor(0, 183, 195, 0))
        p.fillPath(path, QBrush(grad))

        # Line — teal
        pen_line = QPen(QColor(F["teal"]), 2)
        p.setPen(pen_line)
        for i in range(1, len(pts_x)):
            p.drawLine(pts_x[i-1], pts_y[i-1], pts_x[i], pts_y[i])

        p.end()


# =============================================================
#  FLUENT STAT CARD
# =============================================================
class StatCard(QFrame):
    def __init__(self, icon, label, value="—", unit="", accent=F["dv_blue"]):
        super().__init__()
        self._accent = accent
        self.setFixedHeight(96)
        self.setStyleSheet(f"""
            QFrame {{
                background   : {F['surface_card']};
                border-radius: 8px;
                border       : 1px solid {F['border']};
            }}
        """)

        # Fluent acrylic-like left accent strip  via paintEvent
        lo = QVBoxLayout(self)
        lo.setContentsMargins(14, 8, 10, 8)
        lo.setSpacing(2)

        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"color:{accent}; font-size:15px; border:none;")
        lbl = QLabel(label.upper())
        lbl.setStyleSheet(f"color:{F['text_second']}; font-size:10px; font-weight:600; "
                          "letter-spacing:0.8px; border:none;")
        top.addWidget(icon_lbl)
        top.addWidget(lbl)
        top.addStretch()

        self.val_lbl = QLabel(str(value))
        self.val_lbl.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.val_lbl.setStyleSheet(f"color:{F['text_primary']}; border:none;")

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setStyleSheet(f"color:{F['text_disabled']}; font-size:9px; border:none;")

        lo.addLayout(top)
        lo.addWidget(self.val_lbl)
        lo.addWidget(self.unit_lbl)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(0, 8, 4, self.height() - 16, QColor(self._accent))
        p.end()

    def set_value(self, v: str, color=None):
        self.val_lbl.setText(v)
        self.val_lbl.setStyleSheet(
            f"color:{color if color else F['text_primary']}; border:none;"
        )


# =============================================================
#  FLUENT LABEL HELPER
# =============================================================
def fluent_label(text, size=10, bold=False, color=None):
    l = QLabel(text)
    c = color or F["text_second"]
    w = "600" if bold else "400"
    l.setStyleSheet(
        f"color:{c}; font-size:{size}px; font-weight:{w}; "
        "letter-spacing:0.5px; border:none;"
    )
    return l


# =============================================================
#  FLUENT DROP SHADOW HELPER
# =============================================================
def fluent_shadow(widget, blur=12, offset_y=2, alpha=60):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, offset_y)
    shadow.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(shadow)


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
        self._port_refresh_timer = QTimer()
        self._port_refresh_timer.timeout.connect(self._refresh_serial_ports)
        self._port_refresh_timer.start(3000)   # refresh COM list every 3 s
        self._setup_ui()

    # ----------------------------------------------------------
    #  GLOBAL STYLESHEET  — Fluent Light
    # ----------------------------------------------------------
    def _setup_ui(self):
        self.setWindowTitle("Driver Drowsiness Detection System  v3.0")
        self.showMaximized()

        self.setStyleSheet(f"""
            QWidget {{
                background-color : {F['surface_bg']};
                color            : {F['text_primary']};
                font-family      : 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size        : 13px;
            }}
            QLabel  {{ border: none; }}

            /* ── Primary Buttons ── */
            QPushButton {{
                background    : {F['accent']};
                color         : {F['text_on_acc']};
                border        : none;
                border-radius : 4px;
                padding       : 8px 20px;
                font-size     : 13px;
                font-weight   : 600;
            }}
            QPushButton:hover   {{ background: {F['accent_hover']}; }}
            QPushButton:pressed {{ background: {F['accent_press']}; }}
            QPushButton:disabled{{
                background: {F['surface_deep']};
                color     : {F['text_disabled']};
            }}

            /* ── Stop (danger) button ── */
            QPushButton#stop {{
                background  : {F['danger']};
            }}
            QPushButton#stop:hover {{ background: #C50F1F; }}

            /* ── Reset (teal) button ── */
            QPushButton#reset {{
                background  : {F['teal']};
            }}
            QPushButton#reset:hover {{ background: #009AA5; }}

            /* ── Refresh button (outlined) ── */
            QPushButton#refresh {{
                background   : transparent;
                color        : {F['accent']};
                border       : 1px solid {F['accent']};
                border-radius: 4px;
                padding      : 6px 12px;
                font-size    : 12px;
            }}
            QPushButton#refresh:hover {{ background: {F['accent_light']}; }}

            /* ── Inputs ── */
            QComboBox, QLineEdit, QSpinBox {{
                background    : {F['surface_card']};
                border        : 1px solid #C8C6C4;
                border-radius : 4px;
                padding       : 6px 10px;
                color         : {F['text_primary']};
                font-size     : 13px;
                selection-background-color: {F['accent_light']};
            }}
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{
                border: 2px solid {F['accent']};
            }}
            QComboBox::drop-down  {{ border:none; width:24px; }}
            QComboBox::down-arrow {{ width:10px; }}
            QComboBox QAbstractItemView {{
                background          : {F['surface_card']};
                color               : {F['text_primary']};
                selection-background-color: {F['accent_light']};
                selection-color     : {F['text_primary']};
                border              : 1px solid #C8C6C4;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{ width:18px; }}

            /* ── Scroll bars ── */
            QScrollBar:vertical {{ width: 6px; background: transparent; }}
            QScrollBar::handle:vertical {{ background: #C8C6C4; border-radius: 3px; }}
        """)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(16, 12, 16, 12)

        root.addWidget(self._build_header())
        root.addWidget(self._build_control_bar())

        content_lo = QHBoxLayout()
        content_lo.setSpacing(10)

        # Video pane
        video_frame = QFrame()
        video_frame.setStyleSheet(f"""
            QFrame {{
                background   : #000000;
                border-radius: 8px;
                border       : 1px solid {F['border']};
            }}
        """)
        fluent_shadow(video_frame, blur=16, offset_y=3, alpha=40)
        vf_lo = QVBoxLayout(video_frame)
        vf_lo.setContentsMargins(0, 0, 0, 0)
        self._video_lbl = QLabel("NO SIGNAL")
        self._video_lbl.setAlignment(Qt.AlignCenter)
        self._video_lbl.setStyleSheet(
            f"color:{F['text_disabled']}; font-size:20px; font-weight:600; "
            "letter-spacing:4px; background:transparent; border:none;"
        )
        self._video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vf_lo.addWidget(self._video_lbl)
        content_lo.addWidget(video_frame, stretch=7)

        content_lo.addLayout(self._build_right_panel(), stretch=3)
        root.addLayout(content_lo, stretch=1)
        root.addWidget(self._build_footer())

        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._process_latest_frame)

    # ──────────────────────────────────────────────────────────
    def _build_header(self):
        f = QFrame()
        f.setFixedHeight(56)
        f.setStyleSheet(f"""
            QFrame {{
                background   : {F['accent']};
                border-radius: 8px;
            }}
        """)
        fluent_shadow(f)
        lo = QHBoxLayout(f)
        lo.setContentsMargins(20, 0, 20, 0)

        # App icon + title
        icon_lbl = QLabel("◉")
        icon_lbl.setStyleSheet("color:rgba(255,255,255,200); font-size:20px;")

        title = QLabel("Driver Drowsiness Detection")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet("color:#FFFFFF; letter-spacing:0.5px;")

        sub = QLabel("MediaPipe Face Mesh  ·  EAR Algorithm  ·  Real-time Analysis")
        sub.setStyleSheet("color:rgba(255,255,255,160); font-size:11px;")

        lo.addWidget(icon_lbl)
        lo.addSpacing(10)
        lo.addWidget(title)
        lo.addSpacing(16)
        lo.addWidget(sub)
        lo.addStretch()

        # Connection badge
        self._conn_dot = QLabel("●  DISCONNECTED")
        self._conn_dot.setStyleSheet(
            f"color:{F['warning']}; font-size:11px; font-weight:700; "
            "letter-spacing:0.5px; background:rgba(0,0,0,30); "
            "border-radius:4px; padding:4px 10px;"
        )
        lo.addWidget(self._conn_dot)
        return f

    # ──────────────────────────────────────────────────────────
    def _build_control_bar(self):
        f = QFrame()
        f.setStyleSheet(f"""
            QFrame {{
                background   : {F['surface_card']};
                border-radius: 8px;
                border       : 1px solid {F['border']};
            }}
        """)
        fluent_shadow(f, blur=8, offset_y=2, alpha=30)

        lo = QHBoxLayout(f)
        lo.setContentsMargins(18, 10, 18, 10)
        lo.setSpacing(10)

        def sep():
            s = QFrame()
            s.setFrameShape(QFrame.VLine)
            s.setStyleSheet(f"color:{F['border']}; background:{F['border']};")
            s.setFixedWidth(1)
            return s

        # ── Mode ─────────────────────────────────
        lo.addWidget(fluent_label("MODE", bold=True, color=F["text_second"]))
        self._mode_box = QComboBox()
        self._mode_box.addItems([
            "Wireless  (ESP32-CAM HTTP)",
            "Wired  (USB Camera Index)",
            "Serial  (COM Port — ESP32-CAM)",
        ])
        self._mode_box.setFixedWidth(220)
        self._mode_box.currentIndexChanged.connect(self._on_mode_change)
        lo.addWidget(self._mode_box)
        lo.addWidget(sep())

        # ── IP / Index row ────────────────────────
        self._ip_label = fluent_label("IP / INDEX", bold=True, color=F["text_second"])
        lo.addWidget(self._ip_label)
        self._ip_edit = QLineEdit("10.25.251.168")
        self._ip_edit.setFixedWidth(145)
        lo.addWidget(self._ip_edit)

        # ── Port ──────────────────────────────────
        self._port_label = fluent_label("PORT", bold=True, color=F["text_second"])
        lo.addWidget(self._port_label)
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(80)
        self._port_spin.setFixedWidth(80)
        lo.addWidget(self._port_spin)

        # ── Path ──────────────────────────────────
        self._path_label = fluent_label("PATH", bold=True, color=F["text_second"])
        lo.addWidget(self._path_label)
        self._path_edit = QLineEdit("/")
        self._path_edit.setFixedWidth(60)
        lo.addWidget(self._path_edit)

        # ── Serial COM port row (hidden initially) ─
        self._com_label = fluent_label("COM PORT", bold=True, color=F["text_second"])
        self._com_box   = QComboBox()
        self._com_box.setFixedWidth(260)
        self._populate_com_box()

        self._refresh_btn = QPushButton("⟳  Refresh")
        self._refresh_btn.setObjectName("refresh")
        self._refresh_btn.setFixedHeight(32)
        self._refresh_btn.clicked.connect(self._on_refresh_ports)

        lo.addWidget(self._com_label)
        lo.addWidget(self._com_box)
        lo.addWidget(self._refresh_btn)

        # Hide serial widgets initially
        self._com_label.setVisible(False)
        self._com_box.setVisible(False)
        self._refresh_btn.setVisible(False)

        lo.addWidget(sep())

        # ── Action buttons ────────────────────────
        self._btn_connect = QPushButton("▶  Connect")
        self._btn_connect.clicked.connect(self._connect)
        self._btn_connect.setFixedHeight(36)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("stop")
        self._btn_stop.clicked.connect(self._disconnect)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setFixedHeight(36)

        self._btn_reset = QPushButton("↺  Reset")
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

        # ── Alert Banner ──────────────────────────
        self._alert_frame = QFrame()
        self._alert_frame.setFixedHeight(94)
        fluent_shadow(self._alert_frame, blur=12, alpha=50)
        alert_lo = QVBoxLayout(self._alert_frame)
        alert_lo.setContentsMargins(16, 10, 16, 10)
        alert_lo.setSpacing(4)

        al_top = QHBoxLayout()
        self._alert_icon = QLabel("✔")
        self._alert_icon.setFont(QFont("Segoe UI Emoji", 18))
        self._alert_icon.setStyleSheet("border:none;")
        self._alert_lbl = QLabel("ALERT")
        self._alert_lbl.setFont(QFont("Segoe UI", 22, QFont.Bold))
        al_top.addWidget(self._alert_icon)
        al_top.addSpacing(8)
        al_top.addWidget(self._alert_lbl)
        al_top.addStretch()

        self._alert_msg = QLabel("Eyes Open — Driver Alert")
        self._alert_msg.setStyleSheet(f"color:{F['text_second']}; font-size:11px;")
        alert_lo.addLayout(al_top)
        alert_lo.addWidget(self._alert_msg)
        self._apply_alert(LEVEL_NORMAL)
        lo.addWidget(self._alert_frame)

        # ── Stat Cards ────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(8)
        self._card_ear     = StatCard("◎", "EAR",     "—",  "ratio",    F["dv_purple"])
        self._card_perclos = StatCard("⏱", "PERCLOS", "—",  "%",        F["dv_teal"])
        self._card_blinks  = StatCard("👁", "BLINKS",  "0",  "count",   F["dv_blue"])
        self._card_fps     = StatCard("⚡", "FPS",     "—",  "frames/s", F["dv_orange"])
        for c in (self._card_ear, self._card_perclos, self._card_blinks, self._card_fps):
            fluent_shadow(c, blur=8, alpha=30)
        grid.addWidget(self._card_ear,     0, 0)
        grid.addWidget(self._card_perclos, 0, 1)
        grid.addWidget(self._card_blinks,  1, 0)
        grid.addWidget(self._card_fps,     1, 1)
        lo.addLayout(grid)

        # ── EAR Graph ─────────────────────────────
        g_wrap = QFrame()
        g_wrap.setStyleSheet(f"""
            QFrame {{
                background   : {F['surface_card']};
                border-radius: 8px;
                border       : 1px solid {F['border']};
            }}
        """)
        fluent_shadow(g_wrap, blur=8, alpha=25)
        g_lo = QVBoxLayout(g_wrap)
        g_lo.setContentsMargins(12, 10, 12, 10)
        g_lo.setSpacing(4)

        g_header = QHBoxLayout()
        g_lbl = fluent_label("EAR TIME SERIES", bold=True)
        g_thresh = fluent_label("── threshold 0.25", color=F["danger"])
        g_header.addWidget(g_lbl)
        g_header.addStretch()
        g_header.addWidget(g_thresh)
        g_lo.addLayout(g_header)

        self._ear_graph = EARGraph()
        g_lo.addWidget(self._ear_graph)
        lo.addWidget(g_wrap)

        # ── Face Status ───────────────────────────
        self._face_status = QLabel("●  Face Not Detected")
        self._face_status.setAlignment(Qt.AlignCenter)
        self._face_status.setStyleSheet(
            f"color:{F['danger']}; font-size:11px; font-weight:700; "
            f"background:{F['danger_bg']}; border-radius:6px; "
            f"border:1px solid {F['danger']}; padding:5px;"
        )
        lo.addWidget(self._face_status)
        lo.addStretch()
        return lo

    # ──────────────────────────────────────────────────────────
    def _build_footer(self):
        f = QFrame()
        f.setFixedHeight(30)
        f.setStyleSheet(f"""
            QFrame {{
                background   : {F['surface_card']};
                border-radius: 6px;
                border       : 1px solid {F['border']};
            }}
        """)
        lo = QHBoxLayout(f)
        lo.setContentsMargins(14, 0, 14, 0)
        self._status_lbl = fluent_label("System ready — connect a camera to begin.",
                                        color=F["text_second"])
        model_lbl = fluent_label(
            "Model: MediaPipe Face Mesh  |  Algorithm: EAR  |  v3.0",
            color=F["text_disabled"]
        )
        lo.addWidget(self._status_lbl)
        lo.addStretch()
        lo.addWidget(model_lbl)
        return f

    # ----------------------------------------------------------
    #  SERIAL PORT HELPERS
    # ----------------------------------------------------------
    def _populate_com_box(self):
        self._com_box.clear()
        ports = get_serial_ports()
        if ports:
            for dev, desc in ports:
                self._com_box.addItem(f"{dev}  —  {desc}", userData=dev)
        else:
            self._com_box.addItem("No COM ports found", userData=None)

    def _refresh_serial_ports(self):
        """Auto-refresh COM list (called by timer); only rebuilds if count changed."""
        new_ports = get_serial_ports()
        if new_ports != getattr(self, "_last_ports", None):
            self._last_ports = new_ports
            if self._mode_box.currentIndex() == 2:
                self._populate_com_box()

    def _on_refresh_ports(self):
        self._populate_com_box()
        self._status_lbl.setText("Serial port list refreshed.")

    # ----------------------------------------------------------
    #  CAMERA CONTROL
    # ----------------------------------------------------------
    def _on_mode_change(self, idx):
        is_wireless = idx == 0
        is_wired    = idx == 1
        is_serial   = idx == 2

        # Show / hide serial widgets
        self._com_label.setVisible(is_serial)
        self._com_box.setVisible(is_serial)
        self._refresh_btn.setVisible(is_serial)

        # Show / hide IP/port/path widgets
        self._ip_label.setVisible(not is_serial)
        self._ip_edit.setVisible(not is_serial)
        self._port_label.setVisible(is_wireless)
        self._port_spin.setVisible(is_wireless)
        self._path_label.setVisible(is_wireless)
        self._path_edit.setVisible(is_wireless)

        if is_wireless:
            self._ip_edit.setText("10.25.251.168")
            self._ip_edit.setPlaceholderText("10.25.251.168")
        elif is_wired:
            self._ip_edit.setText("0")
            self._ip_edit.setPlaceholderText("Camera index  (0, 1, 2 …)")

        if is_serial:
            self._populate_com_box()

    def _connect(self):
        mode_idx = self._mode_box.currentIndex()

        if mode_idx == 2:  # Serial COM port
            com_dev = self._com_box.currentData()
            if not com_dev:
                self._status_lbl.setText("No COM port selected.")
                return
            source = com_dev
            cap_mode = CaptureThread.MODE_SERIAL
            display_src = f"Serial  {source}"

        elif mode_idx == 0:  # Wireless HTTP
            ip   = self._ip_edit.text().strip()
            port = self._port_spin.value()
            path = self._path_edit.text().strip()
            if not ip.startswith("http"):
                ip = f"http://{ip}"
            source = f"{ip}:{port}{path}"
            cap_mode = CaptureThread.MODE_WIRELESS
            display_src = source

        else:  # Wired USB
            raw = self._ip_edit.text().strip()
            try:
                source = int(raw)
            except ValueError:
                source = 0
            cap_mode = CaptureThread.MODE_WIRED
            display_src = f"Camera index {source}"

        self.detector.reset()
        self._latest_frame = None

        self.capture_thread = CaptureThread(source, mode=cap_mode)
        self.capture_thread.frame_ready.connect(self._on_new_frame)
        self.capture_thread.error.connect(self._on_capture_error)
        self.capture_thread.start()

        self._ui_timer.start(30)

        self._btn_connect.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._conn_dot.setText("●  CONNECTED")
        self._conn_dot.setStyleSheet(
            f"color:#FFFFFF; font-size:11px; font-weight:700; "
            "background:rgba(16,124,16,180); border-radius:4px; padding:4px 10px;"
        )
        self._status_lbl.setText(f"Connected → {display_src}")

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
        self._conn_dot.setStyleSheet(
            f"color:{F['warning']}; font-size:11px; font-weight:700; "
            "background:rgba(0,0,0,30); border-radius:4px; padding:4px 10px;"
        )
        self._status_lbl.setText("Disconnected.")

    def _reset(self):
        self.detector.reset()
        self._status_lbl.setText("Statistics reset.")

    # ----------------------------------------------------------
    #  FRAME PIPELINE
    # ----------------------------------------------------------
    def _on_new_frame(self, frame: np.ndarray):
        self._latest_frame = frame

    def _on_capture_error(self, msg: str):
        self._status_lbl.setText(f"Error: {msg}")
        self._disconnect()

    def _process_latest_frame(self):
        if self._latest_frame is None:
            return

        frame = self._latest_frame.copy()

        now     = time.time()
        elapsed = max(now - self._prev_time, 1e-6)
        self._fps       = int(1.0 / elapsed)
        self._prev_time = now

        frame, ear, face_detected = self.detector.process(frame)
        level   = self.detector.alert_level
        perclos = self.detector.get_perclos()

        # Overlay text using Fluent colours
        overlay_color = {
            LEVEL_NORMAL : (16, 124, 16),
            LEVEL_CAUTION: (0, 120, 215),
            LEVEL_DROWSY : (247, 99, 12),
            LEVEL_DANGER : (232, 17, 35),
        }[level]

        cv2.putText(frame, f"EAR: {ear:.3f}",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, overlay_color, 2)
        cv2.putText(frame, f"FPS: {self._fps}",
                    (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 185, 0), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%",
                    (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (135, 100, 184), 2)

        if level >= LEVEL_DROWSY:
            alert_txt = ALERT_PALETTE[level]["label"]
            cv2.putText(frame, f"!! {alert_txt} !!",
                        (14, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (232, 17, 35) if level == LEVEL_DANGER else (247, 99, 12),
                        3)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img    = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            self._video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._video_lbl.setPixmap(pixmap)

        # Stat cards — map alert level to Fluent status colours
        level_colors = {
            LEVEL_NORMAL : F["success"],
            LEVEL_CAUTION: "#8A5C00",
            LEVEL_DROWSY : F["alert_orange"],
            LEVEL_DANGER : F["danger"],
        }
        c = level_colors[level]
        self._card_ear.set_value(f"{ear:.3f}", c)
        self._card_perclos.set_value(f"{perclos:.1f}", c)
        self._card_blinks.set_value(str(self.detector.blink_count))
        self._card_fps.set_value(str(self._fps))

        self._apply_alert(level)

        if face_detected:
            self._face_status.setText("●  Face Detected")
            self._face_status.setStyleSheet(
                f"color:{F['success']}; font-size:11px; font-weight:700; "
                f"background:{F['success_bg']}; border-radius:6px; "
                f"border:1px solid {F['success']}; padding:5px;"
            )
        else:
            self._face_status.setText("●  Face Not Detected")
            self._face_status.setStyleSheet(
                f"color:{F['danger']}; font-size:11px; font-weight:700; "
                f"background:{F['danger_bg']}; border-radius:6px; "
                f"border:1px solid {F['danger']}; padding:5px;"
            )

        self._ear_graph.push(self.detector.ear_graph)

    # ----------------------------------------------------------
    def _apply_alert(self, level: int):
        p = ALERT_PALETTE[level]
        self._alert_frame.setStyleSheet(f"""
            QFrame {{
                background   : {p['bg']};
                border-radius: 8px;
                border       : 2px solid {p['border']};
            }}
        """)
        self._alert_icon.setText(p["icon"])
        self._alert_lbl.setText(p["label"])
        self._alert_lbl.setStyleSheet(
            f"color:{p['color']}; font-size:22px; font-weight:700; border:none;"
        )
        self._alert_msg.setText(p["msg"])
        self._alert_msg.setStyleSheet(
            f"color:{p['color']}; font-size:11px; opacity:0.8;"
        )

    def closeEvent(self, event):
        self._disconnect()
        self._port_refresh_timer.stop()
        event.accept()


# =============================================================
#  ENTRY POINT
# =============================================================
if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set Fusion palette to light so native widgets (spin boxes etc.) stay light
    palette = QPalette()
    palette.setColor(QPalette.Window,       QColor(F["surface_bg"]))
    palette.setColor(QPalette.WindowText,   QColor(F["text_primary"]))
    palette.setColor(QPalette.Base,         QColor(F["surface_card"]))
    palette.setColor(QPalette.AlternateBase,QColor(F["surface_deep"]))
    palette.setColor(QPalette.Text,         QColor(F["text_primary"]))
    palette.setColor(QPalette.Button,       QColor(F["accent"]))
    palette.setColor(QPalette.ButtonText,   QColor("#FFFFFF"))
    palette.setColor(QPalette.Highlight,    QColor(F["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    win = DrowsinessApp()
    win.show()
    sys.exit(app.exec_())
