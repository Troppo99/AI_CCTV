from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8l.pt")
    model.train(data="D:/MY_FILES/Datasets/two_women/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="two_women")
