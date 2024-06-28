from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8l.pt")
    model.train(data="D:/MY_FILES/Datasets/employee_folding_area.v4i.yolov8/data.yaml", epochs=100, imgsz=738, project=".runs/detect", name="person-l-v3")
