import cv2
import time as t
from datetime import timedelta
from ultralytics import YOLO
import cvzone
import math
import torch
from threading import Thread, Lock


class VideoCaptureThread(Thread):
    def __init__(self, video_path):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(video_path)
        self.ret = False
        self.frame = None
        self.lock = Lock()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.cap.release()


def resize(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height))
    return frame


def last_time(seconds):
    seconds = t.time() - seconds
    print(f"Durasi bertahan selama {timedelta(seconds=int(seconds))}")


def results_elaboration(result, frame, detection_times):
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    for r in result:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                if box.xyxy is None or box.conf is None or box.cls is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                # Draw the bounding box and label
                if conf >= 0.5:
                    # Track the detection time
                    if hasattr(box, "id"):
                        track_id = int(box.id[0])
                        current_time = t.time()

                        if track_id not in detection_times:
                            detection_times[track_id] = current_time
                        else:
                            detection_duration = current_time - detection_times[track_id]
                            duration_text = f"ID {track_id} Time: {timedelta(seconds=int(detection_duration))}"
                            cvzone.putTextRect(frame, duration_text, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    return frame


def main(video_path, model_path):
    # Verifikasi apakah GPU tersedia
    if not torch.cuda.is_available():
        print("CUDA not available. Please check your CUDA installation.")
        return

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    video_thread = VideoCaptureThread(video_path)
    video_thread.start()

    model = YOLO(model_path)
    detection_times = {}
    seconds = t.time()
    mask = cv2.imread(".runs/images/mask2.png")
    if mask is None:
        print("Error: Mask image could not be loaded.")
        return

    try:
        while True:
            try:
                ret, frame = video_thread.get_frame()
                if not ret or frame is None:
                    print("Frame not read properly. Continuing...")
                    continue

                frame_region = cv2.bitwise_and(frame, mask)
                if frame_region is None:
                    print("Frame region not properly defined. Continuing...")
                    continue

                results = model.track(frame_region, persist=True, device="cuda")
                if results is None or not results or all(len(r.boxes) == 0 for r in results):
                    print("No detections found in results. Continuing...")
                    continue

                frame = results_elaboration(results, frame, detection_times)

                cv2.imshow("YOLOv8 Tracking", resize(frame, 0.5))
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    last_time(seconds)
                    break

            except Exception as inner_ex:
                print(f"Inner loop exception: {inner_ex}")
                continue

    except Exception as ex:
        print(f"Exception invalid data: {ex}")
        last_time(seconds)
    finally:
        video_thread.release()
        cv2.destroyAllWindows()
        print("Good luck, Na!")

        # Print duration for each detected ID
        for track_id, start_time in detection_times.items():
            detection_duration = t.time() - start_time
            print(f"ID {track_id} detected for {timedelta(seconds=int(detection_duration))}")


if __name__ == "__main__":
    video_path = ".runs/videos/area_line_tengah.mp4"


    model_path = ".runs/weights/yolov8l.pt"
    main(video_path, model_path)
