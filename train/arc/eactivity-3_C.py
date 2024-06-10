from ultralytics import YOLO


def main():
    data_path = "D:/Punya Waas/Dataset/data.yaml"
    model = YOLO("D:/NWR27/AI_CCTV/.runs/detect/anyar2/weights/last.pt")
    model.train(data=data_path, epochs=100, imgsz=640, project=".runs/detect", name="anyar_C", resume=True)


if __name__ == "__main__":
    main()
