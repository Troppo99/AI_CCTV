from ultralytics import YOLO


def main():
    data_path = "../MY_FILES/Datasets/Employees-3/data.yaml"
    model = YOLO(".runs/detect/Employees-3/weights/last.pt")
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        project=".runs/detect",
        name="Employees-3_C",
        resume=True,
    )


if __name__ == "__main__":
    main()
