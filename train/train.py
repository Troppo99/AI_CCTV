from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8m.pt")
    model.train(data="D:/MY_FILES/Datasets/employee_folding_area.v4i.yolov8/data.yaml", epochs=50, imgsz=736, project=".runs/detect", name="person-m-v2")
