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
    for r in result:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                if box.xyxy is None or box.conf is None or box.cls is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # w, h = x2 - x1, y2 - y1
                conf = math.ceil(box.conf[0] * 100) / 100

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
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    detection_times = {}
    seconds = t.time()
    mask = cv2.imread(".runs/images/mask2.png")

    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Frame not read properly. Continuing...")
                    continue

                frame_region = cv2.bitwise_and(frame, mask)
                if frame_region is None:
                    print("Frame region not properly defined. Continuing...")
                    continue

                results = model.track(frame_region, persist=True)
                if results is None or not results:
                    print("No results from model. Continuing...")
                    continue

                frame = results_elaboration(results, frame, detection_times)

                cv2.imshow("YOLOv8 Tracking", frame)
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
        cap.release()
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
