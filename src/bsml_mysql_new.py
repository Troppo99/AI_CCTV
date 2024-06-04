from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import mysql.connector


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    return cap


def initialize_video_writer(output_path, frame_width, frame_height, fps):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))


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


def main(video_path, output_path, model_people_path, model_activities_path, scale_factor):
    cap = initialize_video_capture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / original_fps)
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

    # Initialize time accumulation dictionary
    time_accumulation = {person: {activity: 0 for activity in class_names_activities} for person in class_names_people}

    # Initialize present status dictionary
    person_present = {person: False for person in class_names_people}

    # Connect to MySQL
    mydb = mysql.connector.connect(host="localhost", user="robot", password="robot123", database="report_ai_cctv", port=3307)
    mycursor = mydb.cursor()

    start_time = time.time()
    frame_counter = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        results_people = model_people(img, stream=True)
        results_activities = model_activities(img, stream=True)

        detections_people = process_detections(results_people, img, class_names_people, 0.8)
        detections_activities = process_detections(results_activities, img, class_names_activities, 0.25)

        # Reset person presence status
        for person in class_names_people:
            person_present[person] = False

        # Combine detections and display them
        for x1, y1, x2, y2, person_class, person_conf in detections_people:
            activity_detected = False
            person_present[person_class] = True
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
                    # Accumulate time for person and activity
                    time_accumulation[person_class][activity_class] += 1 / original_fps
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

        # Display time accumulation at the right edge
        y_position = 15
        absent_persons = []
        for person in class_names_people:
            for activity in time_accumulation[person]:
                time_text = f"{person} {activity} : {format_time(int(time_accumulation[person][activity]))}"
                cvzone.putTextRect(
                    img,
                    time_text,
                    (img.shape[1] - 300, y_position),
                    scale=1,
                    thickness=1,
                    colorT=(255, 255, 255),
                    colorR=(0, 0, 0),
                    colorB=(0, 0, 0),
                    offset=5,
                )
                y_position += 17
            # Check if the person is absent
            if not person_present[person]:
                absent_persons.append(person)

        # Display absent persons
        y_position += 20
        cvzone.putTextRect(
            img,
            "Absent Persons:",
            (img.shape[1] - 300, y_position),
            scale=1,
            thickness=1,
            colorT=(0, 0, 255),
            colorR=(0, 0, 0),
            colorB=(0, 0, 0),
            offset=5,
        )
        y_position += 17
        for person in absent_persons:
            cvzone.putTextRect(
                img,
                person,
                (img.shape[1] - 300, y_position),
                scale=1,
                thickness=1,
                colorT=(0, 0, 255),
                colorR=(0, 0, 0),
                colorB=(0, 0, 0),
                offset=5,
            )
            y_position += 17

        out.write(img)

        # Resize the frame before displaying
        img_resized = cv2.resize(img, new_dim)
        cv2.imshow("Image", img_resized)

        frame_counter += 1
        if frame_counter >= original_fps:
            # Insert data into MySQL every second
            frame_counter = 0
            timestamp = format_time(int(time.time() - start_time))
            for person in class_names_people:
                wrapping_time = format_time(int(time_accumulation[person]["Wrapping"]))
                unloading_time = format_time(int(time_accumulation[person]["unloading"]))
                packing_time = format_time(int(time_accumulation[person]["packing"]))
                sorting_time = format_time(int(time_accumulation[person]["sorting"]))
                absent_person_str = ",".join(absent_persons)

                sql = "INSERT INTO try1 (timestamp, employee_name, wrapping_time, unloading_time, packing_time, sorting_time, absent_person) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                val = (
                    timestamp,
                    person,
                    wrapping_time,
                    unloading_time,
                    packing_time,
                    sorting_time,
                    absent_person_str,
                )
                mycursor.execute(sql, val)
                mydb.commit()

        processing_time = time.time() - start_time
        wait_time = max(1, frame_delay - int(processing_time * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    mydb.close()


if __name__ == "__main__":
    video_path = "../MY_FILES/Videos/CCTV/source/10_ch04_20240425073845.mp4"
    output_path = ".runs/videos/output_video.avi"
    model_people_path = ".runs/detect/.arc/employees-1/weights/best.pt"
    model_activities_path = ".runs/detect/.arc/eactivity-1/weights/best.pt"
    scale_factor = 0.75

    main(video_path, output_path, model_people_path, model_activities_path, scale_factor)
