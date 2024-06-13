import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO(".runs/weights/yolov8l.pt")

video_path = ".runs/videos/tryone.mp4"
cap = cv2.VideoCapture(video_path)

stationary_threshold = 3  # detik
alert_duration = 3  # detik

last_positions = {}
stationary_objects = {}

save_folder = "captured_alerts"
os.makedirs(save_folder, exist_ok=True)


def is_stationary(last_positions, new_positions):
    stationary = {}
    for obj_id, pos in new_positions.items():
        if obj_id in last_positions:
            prev_pos = last_positions[obj_id]
            if np.linalg.norm(np.array(prev_pos) - np.array(pos)) < 5:
                stationary[obj_id] = stationary_objects.get(obj_id, 0) + 1
            else:
                stationary[obj_id] = 0
        else:
            stationary[obj_id] = 0
    return stationary


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    current_positions = {}
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        obj_id = f"{int(cls)}_{x1}_{y1}"
        current_positions[obj_id] = ((x1 + x2) / 2, (y1 + y2) / 2)

    stationary_objects = is_stationary(last_positions, current_positions)
    for obj_id, stationary_time in stationary_objects.items():
        if stationary_time >= stationary_threshold:
            capture_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(save_folder, f"alert_{capture_time}.jpg"), frame)
            break  # Alert sudah diberikan, keluar dari loop

    last_positions = current_positions.copy()
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
