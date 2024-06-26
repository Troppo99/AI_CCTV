from ultralytics import YOLO

model = YOLO(".runs/weights/yolov8l.pt")
model.train(data="", epochs=50, imgsz=640, project=".runs/detect", name="")
