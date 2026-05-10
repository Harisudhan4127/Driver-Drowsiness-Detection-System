from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolov8n.pt")

# =========================
# TRAIN
# =========================
model.train(

    data="data.yaml",

    epochs=100,

    imgsz=640,

    batch=16,

    device=0,

    workers=8,

    cache=True,

    amp=True,

    optimizer="AdamW",

    patience=20,

    project="drowsiness_model",

    name="yolo_train",

    pretrained=True
)

# =========================
# EXPORT
# =========================
model.export(
    format="onnx"
)