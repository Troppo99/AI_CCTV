import cv2
import time as t
from datetime import timedelta
from ultralytics import YOLO
import cvzone
import math
import os


def resize(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height))
    return frame


def last_time(seconds):
    seconds = t.time() - seconds
    print(f"Durasi bertahan selama {timedelta(seconds=int(seconds))}")


def results_elaboration(result, frame, conf_th, detection_times, frame_count, save_folder):
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
    duration_text = ""
    person_detected = False
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            if conf >= conf_th and classNames[cls] == "person":
                cvzone.cornerRect(frame, (x1, y1, w, h))
                cvzone.putTextRect(frame, f"{classNames[cls]}", (max(0, x1), max(35, y1)))

                # Record the time of detection start
                if "person" not in detection_times:
                    detection_times["person"] = t.time()

                person_detected = True
                # Calculate the duration of detection
                duration = t.time() - detection_times["person"]
                duration_text = f"Duration: {timedelta(seconds=int(duration))}"

                # Blink ALERTING! text and save frame only once
                if duration >= 3:
                    if frame_count % 20 < 15:  # Blinking technique
                        cvzone.putTextRect(frame, "ALERTING!", (300, 300), scale=2, thickness=3, colorR=(0, 0, 255))
                        if "saved" not in detection_times:
                            # Save the frame when duration is >= 3 and ALERTING! is displayed
                            capture_time = t.strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(save_folder, f"alert_frame_{capture_time}.jpg")
                            cv2.imwrite(filename, frame)
                            print(f"Frame saved at {capture_time} as {filename}")
                            detection_times["saved"] = True

    # If person is not detected, reset the detection time and remove saved flag
    if not person_detected:
        detection_times.pop("person", None)
        detection_times.pop("saved", None)
        duration_text = "Duration: 0:00:00"

    return duration_text


def main(video_path, model_path, mask_path, conf_th, scale, save_folder):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    seconds = t.time()
    mask = cv2.imread(mask_path)
    detection_times = {}
    frame_count = 0

    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if ret is True:
                frame_region = cv2.bitwise_and(frame, mask)
                results_1 = model(frame_region, stream=True)
                duration_text = results_elaboration(results_1, frame, conf_th, detection_times, frame_count, save_folder)
                cvzone.putTextRect(frame, duration_text, (100, 100))
                frame = resize(frame, scale)
                cv2.imshow("Area Line Tengah", frame)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    last_time(seconds)
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as ex:
        print(f"Exception invalid data: {ex}")
        last_time(seconds)
    finally:
        print("Good luck, Na!")


if __name__ == "__main__":
    video_path = "rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1"
    model_path = ".runs/weights/yolov8l.pt"
    mask_path = ".runs/images/mask2.png"
    save_folder = ".runs/images/alert"

    main(video_path, model_path, mask_path, conf_th=0, scale=0.5, save_folder=save_folder)
