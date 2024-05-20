from ultralytics import YOLO


def main():
    data_path = "D:/NWR27/AI_CCTV/ConstructionSiteSafety/data.yaml"
    model = YOLO("yolov8l.yaml")
    model.train(
        data=data_path, epochs=50, imgsz=640, project="runs/detect", name="train_exp1"
    )


if __name__ == "__main__":
    main()
