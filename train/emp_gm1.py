from ultralytics import YOLO


# def main():
#     data_path = "../MY_FILES/Datasets/emp_gm1/data.yaml"
#     model = YOLO(".runs/weights/yolov8x.pt")
#     model.train(data=data_path, epochs=50, imgsz=640, project=".runs/detect", name="emp_gm1")


if __name__ == "__main__":
    model = YOLO(".runs/weights/yolov8x.pt")
    model.train(data="D:/NWR27/MY_FILES/Datasets/emp_gm1/data.yaml", epochs=50, imgsz=640, project=".runs/detect", name="emp_gm1")
    # main()


# from ultralytics import YOLO
