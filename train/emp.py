from ultralytics import YOLO

model = YOLO(".runs/weights/yolov8x.pt")
model.train(data="D:/NWR27/MY_FILES/Datasets/emp_gm1/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="emp_gm1")
