from ultralytics import YOLO
import cv2

rtsp_url = "rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1"
cap = cv2.VideoCapture(rtsp_url)

model = YOLO("../../MY_FILES/Archives/yang_diignore/.runs/weights/yolov8l.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame, stream=True)

    cv2.imshow("Area Line Tengah", frame)
    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
