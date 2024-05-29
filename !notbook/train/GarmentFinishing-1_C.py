from ultralytics import YOLO


def main():
    model_path = "runs/detect/train_subhanallah/weights/last.pt"
    model = YOLO(model_path)
    model.train(
        data="D:/NWR27/MY_FILES/Datasets/GarmentFinsihing/data.yaml",
        epochs=50,
        imgsz=640,
        project="runs/detect",
        name="train_subhanallah_continued",
        resume=True,
    )


if __name__ == "__main__":
    main()
