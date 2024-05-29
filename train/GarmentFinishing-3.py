from ultralytics import YOLO


def main():
    data_path = "D:/NWR27/MY_FILES/Datasets/GarmentFinishing-3/data.yaml"
    model = YOLO("yolov8l.pt")
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        project="runs/detect",
        name="GarmentFinishing-3",
    )


if __name__ == "__main__":
    main()
