import cv2
import time
from datetime import timedelta
from ultralytics import YOLO
import cvzone
import math


def resize(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height))
    return frame


def last_time(seconds):
    seconds = time.time() - seconds
    print(f"Durasi bertahan selama {timedelta(seconds=int(seconds))}")


def result_elaboration(result, frame, conf_th):
    classNames = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            if conf >= conf_th and classNames[cls]=="mouse":
                cvzone.cornerRect(frame, (x1, y1, w, h))
                cvzone.putTextRect(frame, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)))


def main(video_path, model_path, mask_path, conf_th, scale):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    seconds = time.time()
    mask = cv2.imread(mask_path)
    try:
        while True:
            ret, frame = cap.read()
            if ret is True:
                frame_region = cv2.bitwise_and(frame, mask)
                result_1 = model(frame_region, stream=True)
                result_elaboration(result_1, frame, conf_th)
                frame = resize(frame, scale)
                cv2.imshow("cctv", frame)
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    last_time(seconds)
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as ex:
        print(f"Exception invalid data : {ex}")
        last_time(seconds)
    finally:
        print("Good luck, Na!")


if __name__ == "__main__":
    video_path = ".runs/videos/mouse.mp4"
    model_path = ".runs/weights/yolov8l.pt"
    mask_path = ".runs/images/mask3.png"

    main(video_path, model_path, mask_path, conf_th=0.5, scale=0.75)
