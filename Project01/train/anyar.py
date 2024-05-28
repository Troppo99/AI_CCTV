from ultralytics import YOLO


def main():
    data_path = "D:/Punya Waas/SEMANGAT_1/data.yaml"
    model = YOLO("runs/weights/yolov8l.pt")
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        project="runs/detect",
        name="anyar",
    )


if __name__ == "__main__":
    main()
