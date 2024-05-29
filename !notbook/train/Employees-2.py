from ultralytics import YOLO


def main():
    data_path = "D:/NWR27/MY_FILES/Datasets/EmployeesLatest/data.yaml"
    model = YOLO("runs/weights/yolov8l.pt")
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        project="runs/detect",
        name="train_employees_2",
    )


if __name__ == "__main__":
    main()
