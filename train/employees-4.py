from ultralytics import YOLO


def main():
    data_path = "D:/Punya Waas/Train Person V3/data.yaml"
    model = YOLO(".runs/detect/employees-4/weights/last.pt")
    model.train(data=data_path, epochs=50, imgsz=640, project=".runs/detect", name="employees-4", resume=True)


if __name__ == "__main__":
    main()
