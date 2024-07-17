import torch
import cv2
from ultralytics import YOLO
import math
import cvzone
from datetime import timedelta
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

    def process_frame(self, frame, conf_th, color=(58, 73, 141)):
        def activity(self, frame, conf_th, color=(0, 255, 0)):
            act_boxes_info = []
            results = self.act_model(source=frame, stream=True)
            for r in results:
                for box in r.boxes.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = math.ceil(box.conf[0] * 100) / 100
                    class_id = self.classes[int(box.cls[0])]
                    if conf > conf_th:
                        act_boxes_info.append((x1, y1, x2, y2, class_id, conf, color))
            return frame, act_boxes_info

        results = self.model(source=frame, stream=True)
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

        # if boxes_info:
        #     frame, act_boxes_info = activity(frame, 0)

        return frame, boxes_info

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


class REPORT:
    def __init__(self, classes, backup_file=".runs/data/folding/backup_data.json", data_loaded=True):
        self.classes = classes
        self.data_loaded = data_loaded
        if data_loaded == True:
            self.data = self.load_backup_data(backup_file)
        else:
            self.data = {}
            print(f"Data starts from zero")
        self.last_sent_time = time.time()
        self.backup_file = backup_file

    def update_data(self, emp, existance, frame_duration):
        if emp not in self.data:
            self.data[emp] = {
                "onsite": 0,
                "offsite": 0,
            }
        if existance == "onsite":
            self.data[emp]["onsite"] += frame_duration
        elif existance == "offsite":
            self.data[emp]["offsite"] += frame_duration
        if self.data_loaded == True:
            self.backup_data()

    def backup_data(self):
        with open(self.backup_file, "w") as file:
            json.dump(self.data, file)

    @staticmethod
    def load_backup_data(backup_file):
        if os.path.exists(backup_file):
            with open(backup_file, "r") as file:
                data = json.load(file)
            print(f"Data loaded from {backup_file}")
            return data
        else:
            return {}

    def draw_report(self, frame, toogle=False):
        def format_time(seconds):
            return str(timedelta(seconds=int(seconds)))

        header = ["EMPLOYEE", "ONSITE TIME", "OFFSITE TIME"]
        cv2.putText(frame, header[0], (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, header[1], (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, header[2], (470, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y0, dy = 120, 30
        for i, (emp, times) in enumerate(self.data.items()):
            y = y0 + i * dy
            if emp == "Deti" or emp == "Fifi" or emp == "Nina" or emp == "Hutizah" or emp == "Tia":
                color_row = (200, 30, 0)
            else:
                color_row = (0, 0, 0)
            text_emp = f"  {emp}"
            arrow = ">"
            time_on = format_time(times["onsite"])
            time_off = format_time(times["offsite"])
            if toogle:
                text_on = f"{times['onsite']*100/28800:.2f}%"
                text_off = f"{times['offsite']*100/28800:.2f}%"
            else:
                text_on = f"{time_on}"
                text_off = f"{time_off}"
            cv2.putText(frame, arrow, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA)
            cv2.putText(frame, text_emp, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_row, 2, cv2.LINE_AA)
            cv2.putText(frame, text_on, (299, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_row, 2, cv2.LINE_AA)
            cv2.putText(frame, text_off, (510, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_row, 2, cv2.LINE_AA)

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
                INSERT INTO {table} (cam, timestamp, employee_name, onsite_time, offsite_time)
                VALUES (%s, %s, %s, %s, %s)
                """
                values = ("ROBOTIC ROOM", time.strftime("%Y-%m-%d %H:%M:%S"), emp, times["onsite"], times["offsite"])
                cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            self.last_sent_time = current_time
