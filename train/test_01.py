from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8l.pt")
    model.train(data="C:/Users/Troppo/Downloads/employee_folding_area.v1i.yolov8/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="test_01")
