import cv2
import time
from datetime import timedelta


def format_time(waktu):
    return str(timedelta(seconds=int(waktu)))


video_path = "rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1"
cap = cv2.VideoCapture(video_path)


try:
    seconds = time.time()
    while True:
        ret, frame = cap.read()
        cv2.imshow("cctv", frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            seconds = time.time() - seconds
            print(f"Durasi bertahan selama {format_time(seconds)}") 
            break
    cap.release()
    cv2.destroyAllWindows()
except:
    seconds = time.time() - seconds
    print(f"Durasi bertahan selama {format_time(seconds)}").
    
finally:
    print("Good luck!")