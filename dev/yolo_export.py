from ultralytics import YOLO

model = YOLO(".runs/weights/yolov8l-seg.pt")
results = model(source=".runs/videos/mouse.mp4", show=True, conf=0.5, save=True)
