import cv2
from ultralytics import YOLO

model = YOLO(".runs/weights/yolov8l.pt")

video_path = ".runs/videos/area_line_tengah.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
