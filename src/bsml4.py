import torch
import cv2
from ultralytics import YOLO
import math
import cvzone
from datetime import datetime, timedelta
import os
import json
import time
import pymysql
import numpy as np


class AICCTV:
    def __init__(self, model_path, act_model_path, classes, act_classes, video_path, host):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.act_model = YOLO(act_model_path)
        self.classes = classes
        self.act_classes = act_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Sending to: {host}")

    def process_frame(self, frame, conf_th, mask, color=(58, 73, 141)):
        frame_region = cv2.bitwise_and(frame, mask)

        def activity(frame, color=(0, 255, 0)):
            act_boxes_info = []
            results = self.act_model(source=frame_region, stream=True)
            for r in results:
                for box in r.boxes.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = math.ceil(box.conf[0] * 100) / 100
                    class_id = self.act_classes[int(box.cls[0])]
                    if conf > 0:
                        act_boxes_info.append((x1, y1, x2, y2, class_id, conf, color))
            return frame, act_boxes_info

        results = self.model(source=frame_region, stream=True)
        boxes_info = []
        boxes = []
        confidences = []
        labels = []
        for r in results:
            for box in r.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0] * 100) / 100
                class_id = self.classes[int(box.cls[0])]
                if conf > conf_th:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
                    labels.append((class_id, conf))

        if len(boxes) > 0:
            boxes = np.array(boxes)
            confidences = np.array(confidences)

            indices = self.apply_nms(boxes, confidences)

            class_detections = {}
            for i in indices:
                class_id, conf = labels[i]
                if class_id not in class_detections or conf > class_detections[class_id][1]:
                    class_detections[class_id] = (boxes[i], conf)

            for class_id, (box, conf) in class_detections.items():
                x1, y1, x2, y2 = box
                boxes_info.append((x1, y1, x2, y2, class_id, conf, color))

        if boxes_info:
            frame, act_boxes_info = activity(frame)
        else:
            act_boxes_info = []

        return frame, boxes_info, act_boxes_info

    @staticmethod
    def apply_nms(boxes, confidences, iou_threshold=0.4):
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = confidences.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def capture_frame(self, frame_queue):
        while True:
            if not self.cap.isOpened():
                print("Jaringan putus, menunggu sambungan ulang...")
                while not self.cap.isOpened():
                    # Coba sambungkan kembali setiap 1 detik
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.video_path)

            ret, frame = self.cap.read()
            if not ret:
                print("Frame tidak terbaca, menunggu pembacaan ulang...")
                self.cap.release()
                while not ret:
                    # Coba sambungkan kembali setiap 1 detik
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.video_path)
                    ret, frame = self.cap.read()

            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)

    @staticmethod
    def resize_frame(frame, show_scale):
        height = int(frame.shape[0] * show_scale)
        width = int(frame.shape[1] * show_scale)
        return cv2.resize(frame, (width, height))

    @staticmethod
    def draw_label(frame, x1, y1, x2, y2, text="Your Text", color=(0, 255, 0), thickness=2, font_scale=2, font_thickness=2):
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
    def __init__(self, classes, anto_time=3, backup_dir=".runs/data/test/", data_loaded=True):
        self.data_loaded = data_loaded
        self.backup_dir = backup_dir
        self.current_date = datetime.now().strftime("%Y_%m_%d")
        self.backup_file = os.path.join(backup_dir, f"{self.current_date}.json")

        if data_loaded and os.path.exists(self.backup_file):
            self.data = self.load_backup_data(self.backup_file)
        else:
            self.data = {}
            print(f"Data starts from zero")

        self.classes = classes
        self.anto_time = anto_time
        self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in classes}
        self.last_sent_time = time.time()

    def update_data(self, emp, act, frame_duration):
        current_date = datetime.now().strftime("%Y_%m_%d")

        if current_date != self.current_date:
            self.current_date = current_date
            self.backup_file = os.path.join(self.backup_dir, f"{self.current_date}.json")
            self.reset_data()

        if emp not in self.data:
            self.data[emp] = {
                "folding": 0,
                "idle": 0,
                "offsite": 0,
            }
        if act == "folding":
            self.data[emp]["folding"] += frame_duration
            self.anomaly_tracker[emp]["idle"] = 0
            self.anomaly_tracker[emp]["offsite"] = 0
        elif act == "idle":
            self.anomaly_tracker[emp]["idle"] += frame_duration
            if self.anomaly_tracker[emp]["idle"] > self.anto_time:
                self.data[emp]["idle"] += frame_duration
        elif act == "offsite":
            self.anomaly_tracker[emp]["offsite"] += frame_duration
            if self.anomaly_tracker[emp]["offsite"] > self.anto_time:
                self.data[emp]["offsite"] += frame_duration

        if self.data_loaded:
            self.backup_data()

    def backup_data(self):
        with open(self.backup_file, "w") as file:
            json.dump(self.data, file)

    @staticmethod
    def load_backup_data(backup_file):
        with open(backup_file, "r") as file:
            data = json.load(file)
        print(f"Data loaded from {backup_file}")
        return data

    def reset_data(self):
        self.data = {emp: {"folding": 0, "idle": 0, "offsite": 0} for emp in self.classes}
        self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in self.classes}
        self.backup_data()
        print("Data has been reset due to day change")

    def draw_report(self, frame, toogle=False):
        def format_time(seconds):
            return str(timedelta(seconds=int(seconds)))

        x_move = 2280
        y_move = 1065
        y0, dy = 120, 40
        for i, (emp, times) in enumerate(self.data.items()):
            y = y0 + i * dy
            text_emp = f"{emp}"
            time_folding = format_time(times["folding"])
            time_idle = format_time(times["idle"])
            time_offsite = format_time(times["offsite"])
            if toogle:
                text_folding = f"{times['folding']*100/28800:.1f}%"
                text_idle = f"{times['idle']*100/28800:.1f}%"
                text_offsite = f"{times['offsite']*100/28800:.1f}%"
            else:
                text_folding = f"{time_folding}"
                text_idle = f"{time_idle}"
                text_offsite = f"{time_offsite}"

            cv2.putText(frame, text_emp, (110 + x_move, y + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 217, 165), 2, cv2.LINE_AA)
            cv2.putText(frame, text_folding, (269 + x_move, y + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (124, 225, 143), 2, cv2.LINE_AA)
            cv2.putText(frame, text_idle, (390 + x_move + 35, y + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (101, 224, 225), 2, cv2.LINE_AA)
            cv2.putText(frame, text_offsite, (511 + x_move + 70, y + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (101, 101, 225), 2, cv2.LINE_AA)

    @staticmethod
    def draw_overlay(frame, graphic, x_offset, y_offset):
        graphic_height, graphic_width = graphic.shape[:2]

        if graphic.shape[2] == 4:
            graphic_bgr = graphic[:, :, :3]
            alpha_channel = graphic[:, :, 3]
            mask = alpha_channel / 255.0
            inverse_mask = 1.0 - mask
            for c in range(0, 3):
                frame[y_offset : y_offset + graphic_height, x_offset : x_offset + graphic_width, c] = mask * graphic_bgr[:, :, c] + inverse_mask * frame[y_offset : y_offset + graphic_height, x_offset : x_offset + graphic_width, c]
        else:
            frame[y_offset : y_offset + graphic_height, x_offset : x_offset + graphic_width] = graphic

        return frame

    @staticmethod
    def server_address(host):
        if host == "localhost":
            user = "root"
            password = "robot123"
            database = "report_ai_cctv"
            port = 3306
        elif host == "10.5.0.2":
            user = "robot"
            password = "robot123"
            database = "report_ai_cctv"
            port = 3307
        return user, password, database, port

    def send_data(self, host, user, password, database, port, table):
        current_time = time.time()
        if current_time - self.last_sent_time >= 10:
            conn = pymysql.connect(host=host, user=user, password=password, database=database, port=port)
            cursor = conn.cursor()
            for emp, times in self.data.items():
                query = f"""
                INSERT INTO {table} (cam, timestamp, employee_name, working_time, idle_time, offsite_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = ("FOLDING", time.strftime("%Y-%m-%d %H:%M:%S"), emp, times["folding"], times["idle"], times["offsite"])
                cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            self.last_sent_time = current_time
