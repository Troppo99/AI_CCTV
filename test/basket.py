from ultralytics import YOLO

model = YOLO(".runs/detect/basket/weights/best.pt")
results = model(source="", show=True, conf=0, save=True)
