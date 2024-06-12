import cv2
import time
from datetime import timedelta


def resize(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height))
    return frame


video_path = "rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1"
cap = cv2.VideoCapture(video_path)
scale = 0.75

try:
    seconds = time.time()
    while True:
        ret, frame = cap.read()

        frame = resize(frame, scale)
        cv2.imshow("Area Folding", frame)

        cv2.imshow("cctv", frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            seconds = time.time() - seconds
            print(f"Durasi bertahan selama {timedelta(seconds=int(seconds))}")
            break
    cap.release()
    cv2.destroyAllWindows()
except:
    seconds = time.time() - seconds
    print(f"Durasi bertahan selama {timedelta(seconds=int(seconds))}")

finally:
    print("Good luck!")
