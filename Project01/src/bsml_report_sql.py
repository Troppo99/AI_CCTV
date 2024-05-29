from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import mysql.connector
from mysql.connector import Error


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    return cap


def initialize_video_writer(output_path, frame_width, frame_height, fps):
    return cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height)
    )


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


def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02}:{mins:02}:{secs:02}"


def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            database="report_ai_cctv",
            user="root",
            password="robot123",
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None


def id_zero(cursor):
    id_back_to_zero = """
    ALTER TABLE detail_activity AUTO_INCREMENT = 1;
    """
    cursor.execute(id_back_to_zero)


def insert_activity_data(cursor, timestamp, person_activity_times, absent_person):
    for employee_name, activity_times in person_activity_times.items():
        total_time = sum(activity_times.values())
        percentages = {
            activity: (time / total_time) * 100 if total_time > 0 else 0
            for activity, time in activity_times.items()
        }

        cursor.execute(
            """
            INSERT INTO detail_activity (
                timestamp, employee_name, wrapping_time, wrapping_percentage, 
                unloading_time, unloading_percentage, packing_time, packing_percentage, 
                sorting_time, sorting_percentage, absent_person
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                timestamp,
                employee_name,
                activity_times.get("wrapping", 0),
                percentages.get("wrapping", 0),
                activity_times.get("unloading", 0),
                percentages.get("unloading", 0),
                activity_times.get("packing", 0),
                percentages.get("packing", 0),
                activity_times.get("sorting", 0),
                percentages.get("sorting", 0),
                absent_person,
            ),
        )


def main(
    video_path, output_path, model_people_path, model_activities_path, scale_factor
):
    connection = connect_to_db()
    if connection is None:
        return

    cursor = connection.cursor()
    id_zero(cursor)

    cap = initialize_video_capture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(800 / original_fps)
    new_dim = (int(frame_width * scale_factor), int(frame_height * scale_factor))

    out = initialize_video_writer(output_path, frame_width, frame_height, original_fps)

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

    time_accumulation = {
        person: {activity.lower(): 0 for activity in class_names_activities}
        for person in class_names_people
    }

    last_timestamp = time.time()

    while True:
        start_time = time.time()
        success, img = cap.read()
        if not success:
            print("...computer vision is complete or fail...")
            break

        results_people = model_people(img, stream=True)
        results_activities = model_activities(img, stream=True)

        detections_people = process_detections(
            results_people, img, class_names_people, 0.8
        )
        detections_activities = process_detections(
            results_activities, img, class_names_activities, 0.25
        )

        # Reset person_activity_times for the current second
        person_activity_times = {
            person: {activity.lower(): 0 for activity in class_names_activities}
            for person in class_names_people
        }

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
                if (x1 <= ax1 <= x2 and y1 <= ay1 <= y2) or (
                    x1 <= ax2 <= x2 and y1 <= ay2 <= y2
                ):
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
                    person_activity_times[person_class][activity_class.lower()] += (
                        1 / original_fps
                    )
                    break
            if not activity_detected:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cvzone.putTextRect(
                    img,
                    f"{person_class}",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 0),
                    colorR=(255, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )

        # Insert data into database every second
        if time.time() - last_timestamp >= 1:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            absent_person = ",".join(
                [
                    p
                    for p in class_names_people
                    if p not in [d[4] for d in detections_people]
                ]
            )
            insert_activity_data(
                cursor, timestamp, person_activity_times, absent_person
            )
            connection.commit()
            last_timestamp = time.time()

        out.write(img)

        img_resized = cv2.resize(img, new_dim)
        cv2.imshow("Image", img_resized)

        processing_time = time.time() - start_time
        wait_time = max(1, frame_delay - int(processing_time * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cursor.close()
    connection.close()


if __name__ == "__main__":
    video_path = "../MY_FILES/Videos/CCTV/Train/10_ch04_20240425073845.mp4"
    output_path = "runs/videos/output_video.avi"
    model_people_path = "runs/detect/train_employees/weights/best.pt"
    model_activities_path = "runs/detect/train_subhanallah/weights/best.pt"
    scale_factor = 0.75
    id = 0

    main(
        video_path, output_path, model_people_path, model_activities_path, scale_factor
    )
