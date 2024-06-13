import cv2
from ultralytics import YOLO

model = YOLO(".runs/weights/yolov8l-seg.pt")

video_path = ".runs/videos/mouse.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        seg_frame = results[0].plot()
        cv2.imshow("Mouse", seg_frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
