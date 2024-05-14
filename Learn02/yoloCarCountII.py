from ultralytics import YOLO
import cv2
import cvzone
import math
import sort
import numpy as np

cap = cv2.VideoCapture("Learn02/Videos/Murtaza/cars.mp4")

# Use yolov8l.pt for more stable but lag if don't use a GPU
model = YOLO("Learn02/Yolo-Weights/yolov8L.pt")
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
mask = cv2.imread("Learn02/Images/mask2.png")

# Tracking
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# looping
while True:
    succes, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (
                currentClass == "car"
                or currentClass == "bus"
                or currentClass == "truck"
                or currentClass == "motorbike"
                and conf > 0.5
            ):
                cvzone.putTextRect(
                    img,
                    f"{currentClass} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=1.5,
                    thickness=2,
                    colorT=(25, 50, 100),
                    colorR=(0, 255, 255),
                )
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        print(result)
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(0)
