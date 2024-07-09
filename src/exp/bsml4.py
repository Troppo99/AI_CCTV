import torch
import cv2
from ultralytics import YOLO
import math
import cvzone
from datetime import timedelta
import numpy as np
import pymysql
import time


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


class REPORT:
    def __init__(self, emp_classes, anto_time, interval_send=1):
        self.data = {}
        self.emp_classes = emp_classes
        self.anto_time = anto_time
        self.anomaly_tracker = {emp_class: {"idle_time": 0, "offsite_time": 0} for emp_class in emp_classes}
        self.interval_send = interval_send
        self.last_sent_time = time.time()

    def update_data(self, emp_class, act_class):
        if emp_class not in self.data:
            self.data[emp_class] = {
                "working_time": 0,
                "idle_time": 0,
                "offsite_time": 0,
            }

        # Reset all times to 0 for current second
        self.data[emp_class]["working_time"] = 0
        self.data[emp_class]["idle_time"] = 0
        self.data[emp_class]["offsite_time"] = 0

        if act_class == "working_time":
            self.data[emp_class]["working_time"] = 1
            self.anomaly_tracker[emp_class]["idle_time"] = 0
            self.anomaly_tracker[emp_class]["offsite_time"] = 0
        elif act_class == "idle_time":
            self.anomaly_tracker[emp_class]["idle_time"] += 1
            self.anomaly_tracker[emp_class]["offsite_time"] = 0
            if self.anomaly_tracker[emp_class]["idle_time"] > self.anto_time:
                self.data[emp_class]["idle_time"] = 1
        elif act_class == "offsite_time":
            self.anomaly_tracker[emp_class]["offsite_time"] += 1
            self.anomaly_tracker[emp_class]["idle_time"] = 0
            if self.anomaly_tracker[emp_class]["offsite_time"] > self.anto_time:
                self.data[emp_class]["offsite_time"] = 1

        # Ensure that at least one time is set to 1
        if self.data[emp_class]["working_time"] == 0 and self.data[emp_class]["idle_time"] == 0 and self.data[emp_class]["offsite_time"] == 0:
            if self.anomaly_tracker[emp_class]["idle_time"] > self.anto_time:
                self.data[emp_class]["idle_time"] = 1
            elif self.anomaly_tracker[emp_class]["offsite_time"] > self.anto_time:
                self.data[emp_class]["offsite_time"] = 1
            else:
                self.data[emp_class]["offsite_time"] = 1  # Default to offsite if no condition met

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


def capture_frame(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.qsize() >= 10:
            frame_queue.get()
        frame_queue.put(frame)
