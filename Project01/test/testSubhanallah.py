from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("../MY_FILES/Videos/CCTV/Test/Folding Area_proses sortir.mp4")
cap = cv2.VideoCapture("../MY_FILES/Videos/CCTV/Train/9_ch04_20240424202251.mp4")

model = YOLO("runs/detect/train_subhanallah/weights/best.pt")

classNames = ["Wrapping", "unloading", "packing", "sorting"]

# dimensions of imshow
scaleof = 0.75  # 0 to 1.5 (1280, 720 default)
newDim = (int(1280 * scaleof), int(720 * scaleof))

while True:
    succes, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))
            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.25:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 255),
                    colorR=(0, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )
    img = cv2.resize(img, newDim)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
