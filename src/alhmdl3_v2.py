import cv2
import time
from datetime import timedelta
from ultralytics import YOLO
import cvzone
import math


def resize(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height))
    return frame


def last_time(seconds):
    seconds = time.time() - seconds
    print(f"Durasi bertahan selama {timedelta(seconds=int(seconds))}")


def result_elaboration(result, frame):
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil(box.conf[0] * 100) / 100
            if conf >= 0.5:
                cvzone.putTextRect(frame, f"waiting for duration", (max(0, x1), max(35, y1)))


def main(video_path, scale, model_path, mask_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    seconds = time.time()
    mask = cv2.imread(mask_path)
    prev_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if ret is True:
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                frame_region = cv2.bitwise_and(frame, mask)
                result_1 = model(frame_region, stream=True)
                result_elaboration(result_1, frame)

                # Menampilkan FPS pada frame
                cvzone.putTextRect(frame, f"FPS: {fps:.0f}", (10, 30))

                frame = resize(frame, scale)
                cv2.imshow("cctv", frame)
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    last_time(seconds)
                    break
            else:
                last_time(seconds)
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as ex:
        print(f"Exception invalid data : {ex}")
        last_time(seconds)
    finally:
        print("Good luck, Na!")


if __name__ == "__main__":
    video_path = "rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1"
    video_path = ".runs/videos/area_line_tengah.mp4"
    scale = 0.75
    model_path = ".runs/weights/yolov8l.pt"
    mask_path = ".runs/images/mask2.png"

    main(video_path, scale, model_path, mask_path)
