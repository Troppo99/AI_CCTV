from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)

model = YOLO("Learn02/Yolo-Weights/ppe.pt")

classNames = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

myColor = (0, 0, 255)

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
            if conf > 0.5:
                if (
                    currentClass == "NO-Hardhat"
                    or currentClass == "NO-Safety Vest"
                    or currentClass == "NO-Mask"
                ):
                    myColor = (0, 0, 255)
                elif (
                    currentClass == "Hardhat"
                    or currentClass == "Safety Vest"
                    or currentClass == "Mask"
                ):
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=0.5,
                    thickness=1,
                    colorT=(255, 255, 255),
                    colorR=myColor,
                    colorB=myColor,
                    offset=5,
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
