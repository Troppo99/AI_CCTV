from ultralytics import YOLO
import cv2
import cvzone
import math
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    return cap


def process_detections(results, class_names, conf_th):
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = class_names[cls]
            if conf > conf_th:
                detections.append((x1, y1, x2, y2, currentClass, conf))
    return detections


def update_data_table(data, emp_class, act_class, frame_duration):
    if emp_class not in data:
        data[emp_class] = {
            "wrapping_time": 0,
            "unloading_time": 0,
            "packing_time": 0,
            "sorting_time": 0,
            "idle_time": 0,
            "absent_time": 0,
        }
    if act_class in data[emp_class]:
        data[emp_class][act_class] += frame_duration
    else:
        data[emp_class]["idle_time"] += frame_duration


def calculate_percentages(data):
    percentages = {}
    for emp_class in data:
        total_emp_time = sum(data[emp_class].values())
        percentages[emp_class] = {}
        for key in data[emp_class]:
            percentage_key = f"%{key[0]}"
            if total_emp_time > 0:
                percentages[emp_class][percentage_key] = (data[emp_class][key] / total_emp_time) * 100
            else:
                percentages[emp_class][percentage_key] = 0
    return percentages


def draw_table(img, data, percentages, row_height=25):
    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    x_move = 1250
    y_move = 150
    cv2.putText(img, f"Report Table", (20 + x_move, 540 + y_move), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)
    headers = ["Employee", "Working", "Unloading", "Packing", "Sorting", "Idle", "Absent"]

    scale_text = 1.2
    cvzone.putTextRect(img, headers[0], (20 + x_move, 595 + y_move), scale=scale_text, thickness=1, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[1], (138 + x_move, 595 + y_move), scale=scale_text, thickness=1, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[5], (300 + x_move, 595 + y_move), scale=scale_text, thickness=1, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[6], (460 + x_move, 595 + y_move), scale=scale_text, thickness=1, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))

    pink_color = (255, 0, 255)
    dpink_color = (145, 0, 145)
    for row_idx, (emp_class, times) in enumerate(data.items(), start=1):
        color_rect = pink_color if (row_idx % 2) == 0 else dpink_color
        y_position = 600 + row_idx * row_height
        working_time = times["wrapping_time"] + times["unloading_time"] + times["packing_time"] + times["sorting_time"]
        working_percentages = percentages[emp_class]["%w"] + percentages[emp_class]["%u"] + percentages[emp_class]["%p"] + percentages[emp_class]["%s"]

        cvzone.putTextRect(img, emp_class, (20 + x_move, y_position + y_move), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(working_time), (138 + x_move, y_position + y_move), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{(working_percentages):.0f}%", (228 + x_move, y_position + y_move), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["idle_time"]), (300 + x_move, y_position + y_move), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%i']:.0f}%", (390 + x_move, y_position + y_move), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["absent_time"]), (460 + x_move, y_position + y_move), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%a']:.0f}%", (550 + x_move, y_position + y_move), scale=scale_text, thickness=1, offset=5, colorR=color_rect)


def insert_data_to_mysql(cursor, cam, timestamp, emp_class, times):
    query = """
        INSERT INTO empact (cam, timestamp, employee_name, working_time, idle_time, absent_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
    t_working = times["wrapping_time"] + times["unloading_time"] + times["packing_time"] + times["sorting_time"]
    cursor.execute(
        query,
        (cam, timestamp, emp_class, t_working, times["idle_time"], times["absent_time"]),
    )


def main(video_path, model_emp_path, model_act_path, emp_conf_th, act_conf_th, scale):
    conn = mysql.connector.connect(host="localhost", database="report_ai_cctv", user="robot", password="robot123", port=3307)
    if conn.is_connected():
        cursor = conn.cursor()

        cap = initialize_video_capture(video_path)
        model_emp = YOLO(model_emp_path)
        model_act = YOLO(model_act_path)

        class_names_emp = ["Neneng", "Imas", "Euis", "Siti", "Enok", "Puti", "Sausan", "Eti", "Atik", "Imam"]
        class_names_act = ["wrapping", "unloading", "packing", "sorting"]

        data = {}
        total_time = 0
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        show_table = True  # Flag to toggle table display

        last_insert_time = datetime.now()  # Initialize the last insert time

        while True:
            success, img = cap.read()
            if not success:
                break

            results_emp = model_emp(img, stream=True)
            results_act = model_act(img, stream=True)

            detections_emp = process_detections(results_emp, class_names_emp, emp_conf_th)
            detections_act = process_detections(results_act, class_names_act, act_conf_th)

            # Calculate the duration for this frame based on the frame rate
            frame_duration = 1 / frame_rate
            total_time += frame_duration

            for x1, y1, x2, y2, emp_class, emp_conf in detections_emp:
                act_detected = False
                for ax1, ay1, ax2, ay2, act_class, act_conf in detections_act:
                    if (x1 <= ax1 <= x2 and y1 <= ay1 <= y2) or (x1 <= ax2 <= x2 and y1 <= ay2 <= y2):
                        act_detected = True
                        update_data_table(data, emp_class, act_class.lower() + "_time", frame_duration)
                        break
                if not act_detected:
                    update_data_table(data, emp_class, "idle_time", frame_duration)

                # Draw bounding box for employee
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if act_detected:
                    cvzone.putTextRect(img, f"{emp_class} is {act_class}", (max(0, x1), max(35, y1)), scale=2, thickness=2, colorT=(0, 0, 255), colorR=(0, 255, 255), colorB=(0, 252, 0), offset=5)
                else:
                    cvzone.putTextRect(img, f"{emp_class} is idle", (max(0, x1), max(35, y1)), scale=2, thickness=2, colorT=(0, 0, 0), colorR=(255, 255, 255), colorB=(0, 252, 0), offset=5)

            # Assume no detection means the employee is absent
            detected_employees = [emp_class for _, _, _, _, emp_class, _ in detections_emp]
            for emp_class in class_names_emp:
                if emp_class not in detected_employees:
                    update_data_table(data, emp_class, "absent_time", frame_duration)

            percentages = calculate_percentages(data)
            if show_table:
                draw_table(img, data, percentages)

            # Insert data to MySQL every 1 second
            current_time = datetime.now()
            if (current_time - last_insert_time).total_seconds() >= 1:
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                for emp_class, times in data.items():
                    insert_data_to_mysql(cursor, "CAM001", timestamp, emp_class, times)

                # Commit the transaction
                conn.commit()
                last_insert_time = current_time

            # Header
            cameras = ["CAM001", "CAM002", "CAM003"]  # Amount of camera used
            cvzone.putTextRect(img, f"Camera : {cameras[0]}", (1270, 60), scale=4, thickness=2, offset=7, colorR=(0, 0, 0), colorB=(255, 255, 255))
            cvzone.putTextRect(img, f"Timestamp : " + datetime.now().strftime("%Y/%m/%d %H:%M:%S"), (1270, 100), scale=2, thickness=2, offset=4, colorR=(0, 0, 0), colorB=(255, 255, 255))

            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            resized_img = cv2.resize(img, (width, height))
            cv2.imshow("Area Folding", resized_img)

            key = cv2.waitKey(1) & 0xFF
            if key is ord("n"):
                break
            elif key is ord("s"):
                show_table = not show_table

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "../MY_FILES/Videos/CCTV/source/10_ch04_20240425073845.mp4"
    # video_path = "rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1"
    model_emp_path, model_act_path = ".runs/detect/.arc/employees-1/weights/best.pt", ".runs/detect/.arc/eactivity-1/weights/best.pt"
    emp_conf_th, act_conf_th = (0.8, 0.25)
    video_scale = 0.75

    main(video_path, model_emp_path, model_act_path, emp_conf_th, act_conf_th, video_scale)
