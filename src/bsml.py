from ultralytics import YOLO
import cv2
import cvzone
import math


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    return cap


def process_detections(results, img, class_names, confidence_threshold):
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = class_names[cls]
            if conf > confidence_threshold:
                detections.append((x1, y1, x2, y2, currentClass, conf))
    return detections


def main(video_path, model_people_path, model_activities_path):
    cap = initialize_video_capture(video_path)

    model_people = YOLO(model_people_path)
    model_activities = YOLO(model_activities_path)

    class_names_people = [
        "Neneng",
        "Imas",
        "Euis",
        "Siti",
        "Enok",
        "Puti",
        "Sausan",
        "Eti",
        "Atik",
        "Imam",
    ]
    class_names_activities = ["Wrapping", "unloading", "packing", "sorting"]

    while True:
        success, img = cap.read()
        if not success:
            break

        results_people = model_people(img, stream=True)
        results_activities = model_activities(img, stream=True)

        detections_people = process_detections(results_people, img, class_names_people, 0.8)
        detections_activities = process_detections(results_activities, img, class_names_activities, 0.25)

        # Combine detections and display them
        for x1, y1, x2, y2, person_class, person_conf in detections_people:
            activity_detected = False
            for (
                ax1,
                ay1,
                ax2,
                ay2,
                activity_class,
                activity_conf,
            ) in detections_activities:
                if (x1 <= ax1 <= x2 and y1 <= ay1 <= y2) or (x1 <= ax2 <= x2 and y1 <= ay2 <= y2):
                    activity_detected = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cvzone.putTextRect(
                        img,
                        f"{person_class} = {activity_class}",
                        (max(0, x1), max(35, y1)),
                        scale=2,
                        thickness=2,
                        colorT=(0, 0, 255),
                        colorR=(0, 255, 255),
                        colorB=(0, 252, 0),
                        offset=5,
                    )
                    break
            if not activity_detected:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cvzone.putTextRect(
                    img,
                    f"{person_class} is idle",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 0),
                    colorR=(255, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "../MY_FILES/Videos/CCTV/source/10_ch04_20240425073845.mp4"
    model_people_path = ".runs/detect/.arc/employees-1/weights/best.pt"
    model_activities_path = ".runs/detect/.arc/eactivity-1/weights/best.pt"

    main(video_path, model_people_path, model_activities_path)
