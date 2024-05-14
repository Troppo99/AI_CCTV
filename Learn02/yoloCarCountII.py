from ultralytics import YOLO
import cv2
import cvzone
import math
import sort
import numpy as np

# 1280, 720 default dimensions
scaleof = 0.75  # 0 to 1
newDim = (int(1280 * scaleof), int(720 * scaleof))
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

# masking
mask = cv2.imread("Learn02/Images/mask2.png")

# Tracking
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# limits = [423, 297, 673, 297]
limits = [399, 297, 640, 297]
totalCount = []

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
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (
                currentClass == "car"
                or currentClass == "bus"
                or currentClass == "truck"
                or currentClass == "motorbike"
                and conf > 0.5
            ):
                # cvzone.putTextRect(
                #     img,
                #     f"{currentClass} {conf}",
                #     (max(0, x1), max(35, y1)),
                #     scale=1.5,
                #     thickness=2,
                #     colorT=(25, 50, 100),
                #     colorR=(0, 255, 255),
                # )
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=3, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=3, rt=5, colorR=(255, 0, 0))
        cvzone.putTextRect(
            img,
            f"ID: {int(id)}",
            (max(0, x1), max(35, y1)),
            scale=1.5,
            thickness=2,
            offset=3,
        )

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)

    cvzone.putTextRect(img, f"Count: {totalCount}", (50, 50))

    img = cv2.resize(img, newDim)
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(0)
