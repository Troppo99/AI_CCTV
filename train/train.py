from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8m.pt")
    model.train(data="D:/MY_FILES/Datasets/act_gm1_revised 1/data.yaml", epochs=50, imgsz=736, project=".runs/detect", name="fold-m")
