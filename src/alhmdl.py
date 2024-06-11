from ultralytics import YOLO
import cv2
import cvzone
import math


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    return cap


def main(video_path, model_path):
    cap = initialize_video_capture(video_path)
    model = YOLO(model_path)

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
    while True:
        ret, frame = cap.read()
        result = model(frame, stream=True)
        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(frame, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)),scale=1,thickness=1)

        cv2.imshow("Area Line Tengah", frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1"
    model_path = ".runs/weights/yolov8x.pt"
    main(video_path, model_path)
