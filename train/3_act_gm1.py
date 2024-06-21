from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8x.pt")
    model.train(data="D:/NWR27/MY_FILES/Datasets/act_gm1/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="emp_gm1", resume=False)
