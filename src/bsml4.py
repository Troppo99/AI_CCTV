import torch
import cv2
from ultralytics import YOLO
import math
import cvzone
from datetime import timedelta
import numpy as np


class AICCTV:

    def __init__(
        self,
        emp_model_path,
        act_model_path,
        emp_classes,
        act_classes,
        video_path,
    ):
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
    def __init__(self, emp_classes):
        self.data = {}
        self.emp_classes = emp_classes

    def update_data_table(self, emp_class, act_class, frame_duration):
        if emp_class not in self.data:
            self.data[emp_class] = {
                "working_time": 0,
                "idle_time": 0,
                "offsite_time": 0,
            }
        if act_class in self.data[emp_class]:
            self.data[emp_class][act_class] += frame_duration
        else:
            self.data[emp_class]["idle_time"] += frame_duration

    def calculate_percentages(self):
        percentages = {}
        for emp_class in self.data:
            total_emp_time = sum(self.data[emp_class].values())
            percentages[emp_class] = {}
            for key in self.data[emp_class]:
                percentage_key = f"%{key[0]}"
                if total_emp_time > 0:
                    percentages[emp_class][percentage_key] = (self.data[emp_class][key] / total_emp_time) * 100
                else:
                    percentages[emp_class][percentage_key] = 0
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

            columns = [(emp_class, -160), (format_time(times["working_time"]), 90), (f"{percentages[emp_class]['%w']:.0f}%", 285), (format_time(times["idle_time"]), 430), (f"{percentages[emp_class]['%i']:.0f}%", 625), (format_time(times["offsite_time"]), 770), (f"{percentages[emp_class]['%o']:.0f}%", 965)]

            for text, x_pos in columns:
                cvzone.putTextRect(frame, text, (x_pos + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)


class VideoSaver:
    def __init__(self, output_path, frame_width, frame_height, fps=20.0):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()
