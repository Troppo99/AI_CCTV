"""
Program Title: AI CCTV
Author: Nana Wartana (alias Troppo Lungo)
Field: Robotic and Automation
Company: Gistex Garmen Indonesia
Copyright (c) May 2024 Nana Wartana. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the above copyright notice, this list of conditions, 
and the following disclaimer are retained.

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY; without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""

# Include library
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from datetime import timedelta

cap = cv2.VideoCapture(0)
model = YOLO("runs/weights/ppe.pt")
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
formatted_duration = "00:00:00"
notes = ""

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
                "NO-Mask": bereum,
                "Mask": hejo,
            }
            if conf > 0.5 and currentClass in ["NO-Mask", "Mask"]:
                myColor = colors[currentClass]
                notes = ""

                if currentClass == "Mask":
                    notes = ""
                    if not mask_wearing:
                        mask_wearing = True
                        mask_start_time = time.time()
                else:
                    notes = "(time paused)"
                    if mask_wearing:
                        mask_total_duration += time.time() - mask_start_time
                        mask_wearing = False
                if mask_wearing:
                    current_duration = time.time() - mask_start_time
                    total_duration_display = mask_total_duration + current_duration
                else:
                    total_duration_display = mask_total_duration

                formatted_duration = str(timedelta(seconds=int(total_duration_display)))

                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)
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
        f"Mask Time : {formatted_duration} {notes}",
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
