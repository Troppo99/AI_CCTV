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
import queue
import concurrent.futures


class AICCTV:
    def __init__(self, emp_model_path, act_model_path, emp_classes, act_classes, video_path, host):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.model_emp = YOLO(emp_model_path)
        self.model_act = YOLO(act_model_path)
        self.emp_classes = emp_classes
        self.act_classes = act_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Sending to: {host}")

    def process_frame(self, frame, emp_conf_th=0, act_conf_th=0, mask=None):
        if mask is not None and np.any(mask):
            frame = cv2.bitwise_and(frame, mask)

        emp_boxes_info = []
        results_emp = self.model_emp(source=frame, stream=True)
        frame, emp_boxes_info = self.export_results(frame, results_emp, self.emp_classes, (255, 0, 0), emp_conf_th, "emp")

        act_boxes_info = []
        if emp_boxes_info:
            results_act = self.model_act(source=frame, stream=True)
            frame, act_boxes_info = self.export_results(frame, results_act, self.act_classes, (0, 255, 0), act_conf_th, "act")

        return frame, emp_boxes_info, act_boxes_info

    def export_results(self, frame, results, classes, color, conf_th, model_type):
        boxes_info = []
        boxes = []
        confidences = []
        labels = []

        for r in results:
            for box in r.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0] * 100) / 100
                class_id = classes[int(box.cls[0])]
                if conf > conf_th:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
                    labels.append((class_id, conf))

        if model_type == "emp":
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

        elif model_type == "act":
            for box, (class_id, conf) in zip(boxes, labels):
                x1, y1, x2, y2 = box
                boxes_info.append((x1, y1, x2, y2, class_id, conf, color))

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
                print("Jaringan putus, menunggu sambungan ulang...")
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
    def draw_label(frame, x1, y1, x2, y2, text="Your Text", color=(0, 0, 0), thickness=2, font_scale=2, font_thickness=2):
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
    def __init__(self, emp_classes, anto_time=300, backup_file=".runs/data/backup_data.json", load_data=True):
        if load_data == True:
            self.data = self.load_backup_data(backup_file)
        self.emp_classes = emp_classes
        self.anto_time = anto_time
        self.anomaly_tracker = {emp_class: {"idle_time": 0, "offsite_time": 0} for emp_class in emp_classes}
        self.last_sent_time = time.time()
        self.backup_file = backup_file

    def update_data(self, emp_class, act_class, frame_duration):
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

    def draw_report(self, frame, percentages, row_height=42, x_move=2000, y_move=600, pink_color=(255, 0, 255), dpink_color=(145, 0, 145), scale_text=3):
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

            columns = [
                (emp_class, -160),
                (format_time(times["working_time"]), 90),
                (f"{percentages[emp_class]['%t_w']:.0f}%", 285),
                (format_time(times["idle_time"]), 430),
                (f"{percentages[emp_class]['%t_i']:.0f}%", 625),
                (format_time(times["offsite_time"]), 770),
                (f"{percentages[emp_class]['%t_off']:.0f}%", 965),
            ]

            for text, x_pos in columns:
                cvzone.putTextRect(frame, text, (x_pos + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

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
            for emp_class, times in self.data.items():
                query = f"""
                INSERT INTO {table} (cam, timestamp, employee_name, working_time, idle_time, offsite_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = ("FOLDING", time.strftime("%Y-%m-%d %H:%M:%S"), emp_class, times["working_time"], times["idle_time"], times["offsite_time"])
                cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            self.last_sent_time = current_time


def main(emp_model_path, act_model_path, emp_classes, act_classes, video_path, send=False, host=None, table="empact", mask_path=None, show=False, load_data=True):
    aicctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path, host)
    report = REPORT(aicctv.emp_classes, load_data=load_data)

    frame_queue = queue.Queue(maxsize=10)
    frame_rate = aicctv.cap.get(cv2.CAP_PROP_FPS)
    mask = cv2.imread(mask_path) if mask_path is not None else None
    if host:
        send = True
        user, password, database, port = report.server_address(host)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(aicctv.capture_frame, frame_queue)
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_duration = 1 / frame_rate
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) if mask is not None else None

                """ USER CODE BEGIN: RESULTS PROCESSING ------------------------- """
                frame, emp_boxes_info, act_boxes_info = aicctv.process_frame(frame, mask_resized)
                for x1, y1, x2, y2, emp, _, emp_color in emp_boxes_info:
                    act_detected = False
                    for ax1, ay1, ax2, ay2, act_class, _, act_color in act_boxes_info:
                        if aicctv.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                            act_detected = True
                            report.update_data(emp, "working_time", frame_duration)
                            text = f"{emp} is {act_class}"
                            aicctv.draw_label(frame, x1, y1, x2, y2, text, act_color)
                            break
                    if not act_detected:
                        report.update_data(emp, "idle_time", frame_duration)
                        text = f"{emp} is idle"
                        aicctv.draw_label(frame, x1, y1, x2, y2, text, emp_color)

                detected_employees = [emp for _, _, _, _, emp, _, _ in emp_boxes_info]
                for emp in emp_classes:
                    if emp not in detected_employees:
                        report.update_data(emp, "offsite_time", frame_duration)
                """ USER CODE END: RESULTS PROCESSING --------------------------- """

                if show:
                    percentages = report.calculate_percentages()
                    report.draw_report(frame, percentages)
                    frame = aicctv.resize_frame(frame)
                    mask_info = mask_path.split("/")[-1] if mask_path else mask_path
                    data_info = f"Sending to {host}" if send else "Not sending"
                    text_info = [f"Masking: {mask_info}", f"Data: {data_info}"]
                    j = len(text_info) if host else len(text_info) - 1
                    for i in range(j):
                        cv2.putText(frame, text_info[i], (980, 30 + i * 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(f"Folding", frame)
                if send:
                    report.send_data(host, user, password, database, port, table)
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    break

        aicctv.cap.release()
        cv2.destroyAllWindows()


main(
    emp_model_path=".runs/detect/emp-m/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    emp_classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Umi", "Hutizah", "Anjani", "Tia"],
    act_classes=["Working"],
    video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
    mask_path=".runs/images/mask8.png",
    # host="localhost",
    # host="10.5.0.2",
    show=True,
    load_data=False,
)
