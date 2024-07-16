import torch
import cv2
from ultralytics import YOLO
import math
import cvzone
from datetime import timedelta
import numpy as np
import os
import pymysql
import time
import json
import concurrent.futures
import queue


class AICCTV:
    def __init__(self, emp_model_path, act_model_path, emp_classes, act_classes, video_path, server):
        self.cap = cv2.VideoCapture(video_path)
        self.model_emp = YOLO(emp_model_path)
        self.model_act = YOLO(act_model_path)
        self.class_emp = emp_classes
        self.class_act = act_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.server = server
        print(f"Sending data to: {self.server}")

    def process_frame(self, frame, mask):
        if mask is not None and np.any(mask):
            frame_region = cv2.bitwise_and(frame, mask)
        else:
            frame_region = frame
        results_emp = self.model_emp(source=frame_region, stream=True)
        frame, emp_boxes_info = self.process_results(frame, results_emp, self.class_emp, (255, 0, 0))
        act_boxes_info = []
        if emp_boxes_info:
            results_act = self.model_act(source=frame_region, stream=True)
            frame, act_boxes_info = self.process_results(frame, results_act, self.class_act, (0, 255, 0))
        return frame, emp_boxes_info, act_boxes_info

    def process_results(self, frame, results, classes, color):
        boxes_info = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = self.get_coordinates(box)
                conf = self.get_confidence(box)
                class_id = classes[int(box.cls[0])]
                boxes_info.append((x1, y1, x2, y2, class_id, conf, color))
        return frame, boxes_info

    @staticmethod
    def get_coordinates(box):
        x1, y1, x2, y2 = box.xyxy[0]
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def get_confidence(box):
        return math.ceil(box.conf[0] * 100) / 100

    @staticmethod
    def resize_frame(frame, scale=0.4):
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        return cv2.resize(frame, (width, height))

    @staticmethod
    def draw_box(frame, x1, y1, x2, y2, text, color, thickness=2, font_scale=2, font_thickness=2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cvzone.putTextRect(frame, text, (max(0, x1), max(35, y1)), scale=font_scale, thickness=font_thickness)

    @staticmethod
    def is_overlapping(box1, box2):
        x1, y1, x2, y2 = box1
        ax1, ay1, ax2, ay2 = box2
        if x1 < ax2 and x2 > ax1 and y1 < ay2 and y2 > ay1:
            return True
        return False

    def capture_frame(self, frame_queue):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_queue.qsize() >= 10:
                frame_queue.get()
            frame_queue.put(frame)


class REPORT:
    def __init__(self, emp_classes, anto_time, interval_send, backup_file=".runs/data/backup_data.json"):
        self.data = self.load_backup_data(backup_file)
        self.emp_classes = emp_classes
        self.anto_time = anto_time
        self.anomaly_tracker = {emp_class: {"idle_time": 0, "offsite_time": 0} for emp_class in emp_classes}
        self.interval_send = interval_send
        self.last_sent_time = time.time()
        self.backup_file = backup_file

    def update_data_table(self, emp_class, act_class, frame_duration):
        if emp_class not in self.data:
            self.data[emp_class] = {
                "working_time": 0,
                "idle_time": 0,
                "offsite_time": 0,
            }
        if act_class == "working_time":
            self.data[emp_class]["working_time"] += frame_duration
            self.anomaly_tracker[emp_class]["idle_time"] = 0
            self.anomaly_tracker[emp_class]["offsite_time"] = 0
        elif act_class == "idle_time":
            self.anomaly_tracker[emp_class]["idle_time"] += frame_duration
            if self.anomaly_tracker[emp_class]["idle_time"] > self.anto_time:
                self.data[emp_class]["idle_time"] += frame_duration
        elif act_class == "offsite_time":
            self.anomaly_tracker[emp_class]["offsite_time"] += frame_duration
            if self.anomaly_tracker[emp_class]["offsite_time"] > self.anto_time:
                self.data[emp_class]["offsite_time"] += frame_duration

        self.backup_data()

    def calculate_percentages(self):
        percentages = {}
        for emp_class in self.data:
            t_w = self.data[emp_class]["working_time"]
            t_i = self.data[emp_class]["idle_time"]
            t_off = self.data[emp_class]["offsite_time"]
            t_onsite = t_w + t_i
            t_total = t_onsite + t_off
            if t_onsite > 0:
                percentages[emp_class] = {
                    "%t_w": (t_w / t_onsite) * 100,
                    "%t_i": (t_i / t_onsite) * 100,
                }
            else:
                percentages[emp_class] = {"%t_w": 0, "%t_i": 0}
            if t_total > 0:
                percentages[emp_class]["%t_off"] = (t_off / t_total) * 100
            else:
                percentages[emp_class]["%t_off"] = 0
        return percentages

    def draw_table(self, frame, percentages, row_height=42, x_move=2000, y_move=600, pink_color=(255, 0, 255), dpink_color=(145, 0, 145), scale_text=3):
        def format_time(seconds):
            return str(timedelta(seconds=int(seconds)))

        headers = ["Employee", "Working", "Idle", "Offsite"]
        header_positions = [(-160, 595), (90, 595), (430, 595), (770, 595)]
        header_colors = [(255, 255, 255)] * len(headers)

        cv2.putText(frame, "Report Table", (-140 + x_move, 540 + y_move), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.8, (20, 200, 20), 2, cv2.LINE_AA)
        for header, pos, color in zip(headers, header_positions, header_colors):
            cv2.putText(frame, header, (pos[0] + x_move, pos[1] + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

        for row_idx, (emp_class, times) in enumerate(self.data.items(), start=1):
            color_rect = pink_color if (row_idx % 2) == 0 else dpink_color
            y_position = 610 + row_idx * row_height

            columns = [(emp_class, -160), (format_time(times["working_time"]), 90), (f"{percentages[emp_class]['%t_w']:.0f}%", 285), (format_time(times["idle_time"]), 430), (f"{percentages[emp_class]['%t_i']:.0f}%", 625), (format_time(times["offsite_time"]), 770), (f"{percentages[emp_class]['%t_off']:.0f}%", 965)]

            for text, x_pos in columns:
                cvzone.putTextRect(frame, text, (x_pos + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

    def backup_data(self):
        with open(self.backup_file, "w") as file:
            json.dump(self.data, file)

    def send_to_sql(self, host, user, password, database, port, table_sql, camera_id):
        current_time = time.time()
        if current_time - self.last_sent_time >= self.interval_send:
            conn = pymysql.connect(host=host, user=user, password=password, database=database, port=port)
            cursor = conn.cursor()
            for emp_class, times in self.data.items():
                query = f"""
                INSERT INTO {table_sql} (cam, timestamp, employee_name, working_time, idle_time, offsite_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = (camera_id, time.strftime("%Y-%m-%d %H:%M:%S"), emp_class, times["working_time"], times["idle_time"], times["offsite_time"])
                cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            self.last_sent_time = current_time

    @staticmethod
    def where_sql_server(server):
        if server == "10.5.0.2":
            host = "10.5.0.2"
            user = "robot"
            password = "robot123"
            database = "report_ai_cctv"
            port = 3307
        elif server == "10.5.0.3":
            host = "localhost"
            user = "root"
            password = "robot123"
            database = "report_ai_cctv"
            port = 3306
        return host, user, password, database, port

    @staticmethod
    def load_backup_data(backup_file):
        if os.path.exists(backup_file):
            with open(backup_file, "r") as file:
                data = json.load(file)
            print(f"Data loaded from {backup_file}")
            return data
        else:
            return {}


def main(
    emp_model_path=".runs/detect/emp-m/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    emp_classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Umi", "Hutizah", "Anjani", "Tia"],
    act_classes=["Working"],
    video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
    table_sql="empact",
    server="10.5.0.3",
    camera_id="FOLDING",
    mask_path=None,
    send=False,
    show=False,
    interval_send=1,
    anto_time=3,
):
    start_time = time.time()
    aicctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path, server)
    report = REPORT(emp_classes, anto_time, interval_send)
    frame_rate = aicctv.cap.get(cv2.CAP_PROP_FPS)
    frame_queue = queue.Queue()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(aicctv.capture_frame, frame_queue)
        if server:
            send = True
            host, user, password, database, port = report.where_sql_server(server)
        mask = cv2.imread(mask_path) if mask_path is not None else None
        while aicctv.cap.isOpened():
            if not frame_queue.empty():
                frame = frame_queue.get()
                if frame is None:
                    current_time = time.time()
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"- - -\nframe is None, Buddy! When it was {timestamp}")
                    print(f"Program is running for {current_time-start_time:.0f}!\n- - -")
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) if mask is not None else None
                frame_duration = 1 / frame_rate
                frame, emp_boxes_info, act_boxes_info = aicctv.process_frame(frame, mask_resized)
                for x1, y1, x2, y2, emp_class, _, emp_color in emp_boxes_info:
                    act_detected = False
                    for ax1, ay1, ax2, ay2, act_class, _, act_color in act_boxes_info:
                        if aicctv.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                            act_detected = True
                            report.update_data_table(emp_class, "working_time", frame_duration)
                            text = f"{emp_class} is {act_class}"
                            aicctv.draw_box(frame, x1, y1, x2, y2, text, act_color)
                            break
                    if not act_detected:
                        report.update_data_table(emp_class, "idle_time", frame_duration)
                        text = f"{emp_class} is idle"
                        aicctv.draw_box(frame, x1, y1, x2, y2, text, emp_color)
                detected_employees = [emp_class for _, _, _, _, emp_class, _, _ in emp_boxes_info]
                for emp_class in emp_classes:
                    if emp_class not in detected_employees:
                        report.update_data_table(emp_class, "offsite_time", frame_duration)
                if show:
                    percentages = report.calculate_percentages()
                    report.draw_table(frame, percentages)
                frame = aicctv.resize_frame(frame)
                if show:
                    mask_info = mask_path.split("/")[-1] if mask_path else mask_path
                    data_info = f"Sending to {host}" if send else "Not sending"
                    text_info = [
                        f"Tolerance: {anto_time} seconds",
                        f"Masking: {mask_info}",
                        f"Data: {data_info}",
                        f"Interval Send: {interval_send} seconds",
                    ]
                    j = len(text_info) if server else len(text_info) - 1
                    for i in range(j):
                        cv2.putText(frame, text_info[i], (980, 30 + i * 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(f"Folding Area", frame)
                if send:
                    report.send_to_sql(host, user, password, database, port, table_sql, camera_id)
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    break

        aicctv.cap.release()
        cv2.destroyAllWindows()


main(
    mask_path=".runs/images/mask8.png",
    server="10.5.0.2",
    interval_send=10,
    anto_time=300,
    # show=True,
)
