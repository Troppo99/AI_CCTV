from ultralytics import YOLO


def main():
    model_path = "runs/detect/train_exp14/weights/last.pt"
    model = YOLO(model_path)
    model.train(
        data="D:/NWR27/AI_CCTV/ConstructionSiteSafety/data.yaml",
        epochs=50,
        imgsz=640,
        project="runs/detect",
        name="train_exp14_continued",
        resume=True,
    )


if __name__ == "__main__":
    main()
