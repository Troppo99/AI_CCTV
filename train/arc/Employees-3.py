from ultralytics import YOLO


def main():
    data_path = "D:/NWR27/MY_FILES/Datasets/Employees-3/data.yaml"
    model = YOLO(".runs/weights/yolov8l.pt")
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        project=".runs/detect",
        name="Employees-3",
    )


if __name__ == "__main__":
    main()
