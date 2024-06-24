import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO(".runs/detect/two_women/weights/best.pt")
model.fuse()

cap = cv2.VideoCapture("D:/AI_CCTV/.runs/videos/0624.mp4")
assert cap.isOpened()
while cap.isOpened():
    ret, frame = cap.read()
    start_time = time()
    
    results = model(frame)
    xyxys = []
    confidences = []
    class_ids = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            if len(box) == 4:
                xyxys.append(box[:4])
                confidences.append(box[4])
                class_ids.append(int(box[5]))

    end_time = time()
    fps = 1 / (end_time - start_time)

    for i, box in enumerate(xyxys):
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            class_id = class_ids[i]
            label = f"{class_id} {confidence:.2f}"
            cv2.rectangle(results[0].plot(), (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(results[0].plot(), label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(results[0].plot(), f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Detection", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
