import cv2

# RTSP URL of the CCTV camera. Example: rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101
rtsp_url = "rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1"

print("Reading ret...")
cap = cv2.VideoCapture(rtsp_url)
# cap = cv2.VideoCapture("../MY_FILES/Videos/CCTV/source/10_ch04_20240425073845.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed...")
        break
    print("ret is True...")
    cv2.imshow("CCTV Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
