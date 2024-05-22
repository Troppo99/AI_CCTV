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

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import sort
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("../MY_FILES/Videos/CCTV/Train/10_ch04_20240425073845.mp4")

# Initialize YOLO models
model_person = YOLO("runs/weights/yolov8l.pt")  # Model for detecting people
model_custom = YOLO(
    "runs/detect/train_subhanallah/weights/best.pt"
)  # Custom model for detecting specific classes

# Class names for the custom model
classNames = ["Wrapping", "unloading", "packing", "sorting"]

# Dimensions for imshow
scaleof = 0.75  # 0 to 1.5 (1280, 720 default)
newDim = (int(1280 * scaleof), int(720 * scaleof))

# Get frame width, height, and original frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / original_fps)  # Delay in milliseconds

# Initialize VideoWriter object with the original frame rate
out = cv2.VideoWriter(
    "runs/videos/output_video.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    original_fps,
    (frame_width, frame_height),
)

# Initialize the SORT tracker
tracker = sort.Sort(max_age=30, min_hits=3, iou_threshold=0.3)

while True:
    start_time = time.time()
    success, img = cap.read()
    if not success:
        break

    # Detect people using the first model
    results_person = model_person(img, stream=True)
    detections = np.empty((0, 5))  # Initialize detections array

    for r in results_person:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.25 and cls == 0:  # Class 0 is usually 'person' in COCO dataset
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cvzone.putTextRect(
                    img,
                    f"Person {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 255),
                    colorR=(0, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )

    # Update tracker with detections
    resultTracker = tracker.update(detections)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 220, 20), 3)
        cvzone.putTextRect(
            img,
            f"ID: {int(id)}",
            (max(0, x1), max(35, y1)),
            scale=2,
            thickness=2,
            colorT=(0, 0, 255),
            colorR=(0, 255, 0),
            offset=5,
        )

    # Detect custom classes using the second model
    results_custom = model_custom(img, stream=True)

    for r in results_custom:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # if conf > 0.25:
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            #     cvzone.putTextRect(
            #         img,
            #         f"{currentClass} {conf}",
            #         (max(0, x1), max(35, y1)),
            #         scale=2,
            #         thickness=2,
            #         colorT=(0, 0, 255),
            #         colorR=(0, 255, 255),
            #         colorB=(0, 252, 0),
            #         offset=5,
            #     )

    # Write the frame to the video file
    out.write(img)

    # Display the frame
    img_resized = cv2.resize(img, newDim)
    cv2.imshow("Image", img_resized)

    # Calculate processing time and add delay
    processing_time = time.time() - start_time
    wait_time = max(
        1, frame_delay - int(processing_time * 1000)
    )  # Ensure non-negative wait time
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()
