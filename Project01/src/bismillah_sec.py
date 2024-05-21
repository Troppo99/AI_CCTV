# Include library
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from datetime import timedelta

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

# Colors settings
myColor = (128, 128, 128)
bulao = (255, 0, 0)
hejo = (0, 255, 0)
bereum = (0, 0, 255)
koneng = (0, 255, 255)
hideung = (0, 0, 0)
bodas = (255, 255, 255)

# Time settings
mask_wearing = False
mask_start_time = 0
mask_total_duration = 0

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
            colors = {
                "NO-Mask": bereum,  # Redefine 'bereum' as needed
                "Mask": hejo,  # Redefine 'hejo' as needed
            }

            if conf > 0.5 and currentClass in ["NO-Mask", "Mask"]:
                myColor = colors[currentClass]

                if currentClass == "Mask":
                    if not mask_wearing:
                        mask_wearing = True
                        mask_start_time = (
                            time.time()
                        )  # Start timing when mask is first detected
                else:
                    if mask_wearing:
                        mask_total_duration += (
                            time.time() - mask_start_time
                        )  # Accumulate wearing time
                        mask_wearing = False  # Reset mask wearing status

                # Calculate the current or total duration for display
                if mask_wearing:
                    current_duration = time.time() - mask_start_time
                    total_duration_display = mask_total_duration + current_duration
                else:
                    total_duration_display = mask_total_duration

                # Format the duration for display
                formatted_duration = str(timedelta(seconds=int(total_duration_display)))

                # Drawing the rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)

                # Constructing the text to display
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(0, y2 + 15)),
                    scale=1,
                    thickness=1,
                    colorT=koneng,
                    colorR=hideung,
                    colorB=myColor,
                    offset=1,
                )
            """LET'S PLAY WITH YOUR BRAIN"""

    cvzone.putTextRect(
        img,
        f"Mask Time : {formatted_duration}",
        (10, 40),
        scale=2,
        thickness=2,
        colorT=bodas,
        colorR=bulao,
        colorB=hejo,
        offset=3,
    )
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
