# Module
from ultralytics import YOLO
import cv2
import cvzone
import math
import sort

# Source videos (1,6,7,9,10) 7 is best
video_paths = {
    1: "Project01/Datasets/Videos/1_ch04_20240418061050.mp4",  # Awesome
    6: "Project01/Datasets/Videos/6_ch04_20240423113600.mp4",  # not bad
    7: "Project01/Datasets/Videos/7_ch04_20240423200452.mp4",  # Good - Top
    9: "Project01/Datasets/Videos/9_ch04_20240424202251.mp4",  # nice - Top
    10: "Project01/Datasets/Videos/10_ch04_20240425073845.mp4",  # excellent
}
key = 7
video_path = video_paths[key]
cap = cv2.VideoCapture(video_path)

# Pre-trained model (yolov8l or yolov8n)
model = YOLO("Learn02/Yolo-Weights/yolov8l.pt")

# Dimensions
scaleof = 0.75  # 0 to 1.5 (1280, 720 default)
newDim = (int(1280 * scaleof), int(720 * scaleof))

# Tracking
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Label
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
    succes, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
                if currentClass == "person":
                    cRect = (0, 255, 0)
                    cText = (0, 0, 255)
                    cFrame = (100, 220, 20)
                else:
                    cRect = (200, 200, 200)
                    cText = (0, 0, 0)
                    cFrame = (200, 10, 200)

                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=3, rt=5, colorR=cFrame)
                cv2.rectangle(img, (x1, y1), (x2, y2), cFrame, 3)
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=cText,
                    colorR=cRect,
                    offset=5,
                )
    img = cv2.resize(img, newDim)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
