from ultralytics import YOLO

model = YOLO(".runs/detect/emp_gm1/weights/best.pt")
results = model(source="F:Video Dede Uji.mp4", show=True, conf=0.56, save=True)
