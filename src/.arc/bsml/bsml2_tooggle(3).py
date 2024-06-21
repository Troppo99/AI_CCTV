from ultralytics import YOLO
import cv2
import cvzone
import math
from datetime import datetime, timedelta


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
        data[emp_class] = {"wrapping_time": 0, "unloading_time": 0, "packing_time": 0, "sorting_time": 0, "idle_time": 0, "absent_time": 0}
    if act_class in data[emp_class]:
        data[emp_class][act_class] += frame_duration
    else:
        data[emp_class]["idle_time"] += frame_duration


def calculate_percentages(data, total_time):
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


def draw_table(img, data, percentages, row_height=40):
    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    # cvzone.putTextRect(img, f"Report Table", (20, 500), scale=4, thickness=2, offset=7, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cv2.putText(img, f"Report Table", (20, 535), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
    headers = ["Employee", "Wrapping", "Unloading", "Packing", "Sorting", "Idle", "Absent"]  # 7

    # Column
    scale_text = 2
    x_position = [218, 368, 506, 656, 794, 944, 1082, 1232, 1370, 1520, 1658, 1808]
    cvzone.putTextRect(img, headers[0], (20, 600), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[1], (x_position[0], 600), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[2], (x_position[2], 600), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[3], (x_position[4], 600), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[4], (x_position[6], 600), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[5], (x_position[8], 600), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(img, headers[6], (x_position[10], 600), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))

    # Record
    pink_color = (255, 0, 255)
    dpink_color = (145, 0, 145)
    for row_idx, (emp_class, times) in enumerate(data.items(), start=1):
        y_position = 600 + row_idx * row_height
        color_rect = pink_color
        if (row_idx % 2) == 0:
            color_rect = dpink_color
        else:
            color_rect == pink_color
        cvzone.putTextRect(img, emp_class, (20, y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["wrapping_time"]), (x_position[0], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%w']:.0f}%", (x_position[1], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["unloading_time"]), (x_position[2], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%u']:.0f}%", (x_position[3], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["packing_time"]), (x_position[4], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%p']:.0f}%", (x_position[5], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["sorting_time"]), (x_position[6], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%s']:.0f}%", (x_position[7], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["idle_time"]), (x_position[8], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%i']:.0f}%", (x_position[9], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)

        cvzone.putTextRect(img, format_time(times["absent_time"]), (x_position[10], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)
        cvzone.putTextRect(img, f"{percentages[emp_class]['%a']:.0f}%", (x_position[11], y_position), scale=scale_text, thickness=1, offset=5, colorR=color_rect)


def main(video_path, model_emp_path, model_act_path, emp_conf_th, act_conf_th, scale):
    cap = initialize_video_capture(video_path)
    model_emp = YOLO(model_emp_path)
    model_act = YOLO(model_act_path)

    class_names_emp = ["Neneng", "Imas", "Euis", "Siti", "Enok", "Puti", "Sausan", "Eti", "Atik", "Imam"]
    class_names_act = ["wrapping", "unloading", "packing", "sorting"]

    data = {}
    total_time = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    show_table = True  # Flag to toggle table display

    while True:
        success, img = cap.read()
        if not success:
            break
        # start_time = datetime.now() # unused

        # Header
        cameras = ["CAM001", "CAM002", "CAM003"]  # Amount of camera used
        cvzone.putTextRect(img, f"Camera : {cameras[0]}", (20, 60), scale=4, thickness=2, offset=7, colorR=(0, 0, 0), colorB=(255, 255, 255))
        cvzone.putTextRect(img, f"Timestamp : " + datetime.now().strftime("%Y/%m/%d %H:%M:%S"), (20, 100), scale=2, thickness=2, offset=4, colorR=(0, 0, 0), colorB=(255, 255, 255))

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
                cvzone.putTextRect(
                    img,
                    f"{emp_class} is {act_class}",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 255),
                    colorR=(0, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )
            else:
                cvzone.putTextRect(
                    img,
                    f"{emp_class} is idle",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 0),
                    colorR=(255, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )

        # Assume no detection means the employee is absent
        detected_employees = [emp_class for _, _, _, _, emp_class, _ in detections_emp]
        for emp_class in class_names_emp:
            if emp_class not in detected_employees:
                update_data_table(data, emp_class, "absent_time", frame_duration)

        percentages = calculate_percentages(data, total_time)
        if show_table:
            draw_table(img, data, percentages)

        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        resized_img = cv2.resize(img, (width, height))
        cv2.imshow("Image", resized_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            break
        elif key == ord("s"):
            show_table = not show_table

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "../MY_FILES/Videos/CCTV/source/10_ch04_20240425073845.mp4"
    model_emp_path, model_act_path = ".runs/detect/.arc/employees-1/weights/best.pt", ".runs/detect/.arc/eactivity-1/weights/best.pt"
    emp_conf_th, act_conf_th = (0.8, 0.25)
    video_scale = 0.75

    main(video_path, model_emp_path, model_act_path, emp_conf_th, act_conf_th, video_scale)