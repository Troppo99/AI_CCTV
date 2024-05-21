from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
model = YOLO("../MY_FILES/Yolo-Models/ppe.pt")
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
myColor = (128,128,128)
bulao = (255, 0, 0)
hejo = (0, 255, 0)
bereum = (0, 0, 255)
koneng = (0,255,255)
hideung = (0,0,0)
bodas = (255,255,255)

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
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            """LET'S PLAY WITH YOUR BRAIN"""
            read = 0
            if conf > 0.5:
                if currentClass == "NO-Mask":
                    myColor = bereum
                    read = 1
                elif currentClass == "Mask":
                    myColor = hejo
                    read = 1
                if read == 1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)
                    cvzone.putTextRect(
                        img,
                        f"{classNames[cls]} {conf}",
                        (max(0, x1), max(0, y2+15)),
                        scale=1,
                        thickness=1,
                        colorT=koneng,
                        colorR=hideung,
                        colorB=myColor,
                        offset=1,
                    )
            """LET'S PLAY WITH YOUR BRAIN"""

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
