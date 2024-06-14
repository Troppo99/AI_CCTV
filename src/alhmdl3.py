import cv2
import time as t
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
    seconds = t.time() - seconds
    print(f"Durasi bertahan selama {timedelta(seconds=int(seconds))}")


def results_elaboration(result, frame, detection_times):
    classNames = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    for r in result:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                # Draw the bounding box and label
                if conf >= 0.5:
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                    cvzone.putTextRect(frame, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)))

                # Track the detection time
                if hasattr(box, "id"):
                    track_id = int(box.id[0])
                    current_time = t.time()

                    if track_id not in detection_times:
                        detection_times[track_id] = current_time
                    else:
                        detection_duration = current_time - detection_times[track_id]
                        duration_text = f"ID {track_id} Time: {timedelta(seconds=int(detection_duration))}"
                        cv2.putText(frame, duration_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def main(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    detection_times = {}
    seconds = t.time()

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                results = model.track(frame, persist=True)
                frame = results_elaboration(results, frame, detection_times)

                cv2.imshow("YOLOv8 Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    last_time(seconds)
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Print duration for each detected ID
        for track_id, start_time in detection_times.items():
            detection_duration = t.time() - start_time
            print(f"ID {track_id} detected for {timedelta(seconds=int(detection_duration))}")

    except Exception as ex:
        print(f"Exception invalid data: {ex}")
        last_time(seconds)
    finally:
        print("Good luck, Na!")


if __name__ == "__main__":
    video_path = ".runs/videos/area_line_tengah.mp4"
    model_path = ".runs/weights/yolov8l.pt"
    main(video_path, model_path)
