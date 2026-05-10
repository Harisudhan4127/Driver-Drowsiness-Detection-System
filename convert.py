from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.export(

    format="onnx",

    imgsz=320,

    dynamic=True,

    simplify=True,

    opset=12
)