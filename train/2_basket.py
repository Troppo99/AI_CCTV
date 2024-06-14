from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(".runs/detect/basket/weights/last.pt")
    model.train(data="D:/NWR27/MY_FILES/Datasets/basket/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="basket", resume=True)
