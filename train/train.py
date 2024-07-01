from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/detect/person-m-v2/weights/best.pt")
    model.train(data="D:/MY_FILES/Datasets/employee_folding_area.v7i.yolov8/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="emp-m")
