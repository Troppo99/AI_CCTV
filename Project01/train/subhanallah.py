from ultralytics import YOLO


def main():
    data_path = "D:/NWR27/MY_FILES/Datasets/GarmentFinsihing/data.yaml"
    model = YOLO("yolov8l.pt")
    model.train(
        data=data_path, epochs=50, imgsz=640, project="runs/detect", name="train_subhanallah"
    )


if __name__ == "__main__":
    main()
