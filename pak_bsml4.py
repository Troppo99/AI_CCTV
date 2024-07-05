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


class AICCTV:
    def __init__(self, emp_model_path, act_model_path, emp_classes, act_classes, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.model_emp = YOLO(emp_model_path)
        self.model_act = YOLO(act_model_path)
        self.class_emp = emp_classes
        self.class_act = act_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

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


class REPORT:
    def __init__(self, emp_classes, anto_time, interval_send):
        self.data = {}
        self.emp_classes = emp_classes
        self.anto_time = anto_time
        self.anomaly_tracker = {emp_class: {"idle_time": 0, "offsite_time": 0} for emp_class in emp_classes}
        self.interval_send = interval_send
        self.last_sent_time = time.time()

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
        if server == "Waskita":
            host = "192.168.100.187"
            user = "robot"
            password = "robot123"
            database = "report_ai_cctv"
            port = 3307
        elif server == "Nana":
            host = "localhost"
            user = "root"
            password = "robot123"
            database = "report_ai_cctv"
            port = 3306
        return host, user, password, database, port


class SAVER:
    def __init__(self, output_path, frame_width, frame_height, fps=20.0):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()

    @staticmethod
    def uniquifying(base_path, base_name, extension):
        version = 1
        while True:
            filename = f"{base_name}_v{version}{extension}"
            full_path = os.path.join(base_path, filename)
            if not os.path.exists(full_path):
                return full_path
            version += 1


def capture_frame(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.qsize() >= 10:
            frame_queue.get()
        frame_queue.put(frame)
