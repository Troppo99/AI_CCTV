from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8s.pt")
    model.train(data="C:/Users/Troppo/Downloads/employee_folding_area.v2i.yolov8/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="test_02")
