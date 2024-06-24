import torch
import cv2
from ultralytics import YOLO
import math
import cvzone
from datetime import datetime, timedelta


class AICCTV:
    def __init__(self, video_path, mask_path, emp_model_path, act_model_path, emp_classes, act_classes):
        self.cap = cv2.VideoCapture(video_path)
        self.mask = cv2.imread(mask_path)
        self.model_emp = YOLO(emp_model_path)
        self.model_act = YOLO(act_model_path)
        self.class_emp = emp_classes
        self.class_act = act_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def __call__(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame, emp_boxes_info, act_boxes_info = self.process_frame(frame)

            for x1, y1, x2, y2, emp_class, emp_conf, _ in emp_boxes_info:
                for ax1, ay1, ax2, ay2, act_class, act_conf, _ in act_boxes_info:
                    if self.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                        text = f"{emp_class} is {act_class}"
                        self.draw_box(frame, x1, y1, x2, y2, text, (0, 255, 0))

            frame = self.resize_frame(frame)
            cv2.imshow("AI on Folding Area", frame)
            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

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

    def process_frame(self, frame):
        # frame_region = cv2.bitwise_and(frame, self.mask)

        # Process employee detection
        results_emp = self.model_emp(source=frame, stream=True)
        frame, emp_boxes_info = self.process_results(frame, results_emp, self.class_emp, (0, 255, 0))

        # Process activity detection
        act_boxes_info = []
        if emp_boxes_info:  # Only process activities if employees are detected
            results_act = self.model_act(source=frame, stream=True)
            frame, act_boxes_info = self.process_results(frame, results_act, self.class_act, (255, 0, 0))

        return frame, emp_boxes_info, act_boxes_info

    @staticmethod
    def get_coordinates(box):
        x1, y1, x2, y2 = box.xyxy[0]
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def get_confidence(box):
        return math.ceil(box.conf[0] * 100) / 100

    @staticmethod
    def resize_frame(frame, scale=0.6):
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


class REPORT(AICCTV):
    def __init__(self, video_path, mask_path, emp_model_path, act_model_path, emp_classes, act_classes):
        super().__init__(video_path, mask_path, emp_model_path, act_model_path, emp_classes, act_classes)
        self.data = {}
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

    def __call__(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame, emp_boxes_info, act_boxes_info = self.process_frame(frame)
            frame_duration = 1 / self.frame_rate

            for x1, y1, x2, y2, emp_class, _, _ in emp_boxes_info:
                act_detected = False
                for ax1, ay1, ax2, ay2, act_class, _, _ in act_boxes_info:
                    if self.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                        act_detected = True
                        self.update_data_table(emp_class, act_class.lower() + "_time", frame_duration)
                        text = f"{emp_class} is {act_class}"
                        self.draw_box(frame, x1, y1, x2, y2, text, (0, 255, 0))
                        break
                if not act_detected:
                    self.update_data_table(emp_class, "idle_time", frame_duration)
                    text = f"{emp_class} is idle"
                    self.draw_box(frame, x1, y1, x2, y2, text, (255, 255, 0))

            detected_employees = [emp_class for _, _, _, _, emp_class, _, _ in emp_boxes_info]
            for emp_class in self.class_emp:
                if emp_class not in detected_employees:
                    self.update_data_table(emp_class, "offsite_time", frame_duration)

            percentages = self.calculate_percentages(self.data)
            self.draw_table(frame, self.data, percentages)

            frame = self.resize_frame(frame)
            cv2.imshow("AI on Folding Area", frame)
            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_data_table(self, emp_class, act_class, frame_duration):
        if emp_class not in self.data:
            self.data[emp_class] = {
                "folding_time": 0,
                "idle_time": 0,
                "offsite_time": 0,
            }
        if act_class in self.data[emp_class]:
            self.data[emp_class][act_class] += frame_duration
        else:
            self.data[emp_class]["idle_time"] += frame_duration

    def calculate_percentages(self, data):
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

    def draw_table(self, img, data, percentages, row_height=30):
        def format_time(seconds):
            return str(timedelta(seconds=int(seconds)))

        x_move = 1250
        y_move = 270
        cv2.putText(img, f"Report Table", (-160 + x_move, 540 + y_move), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)
        headers = ["Employee", "Folding", "Idle", "Offsite"]

        scale_text = 1.8
        cvzone.putTextRect(img, headers[0], (-160 + x_move, 595 + y_move), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
        cvzone.putTextRect(img, headers[1], (x_move, 595 + y_move), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
        cvzone.putTextRect(img, headers[2], (210 + x_move, 595 + y_move), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))
        cvzone.putTextRect(img, headers[3], (420 + x_move, 595 + y_move), scale=scale_text, thickness=2, offset=5, colorR=(0, 0, 0), colorB=(255, 255, 255))

        pink_color = (255, 0, 255)
        dpink_color = (145, 0, 145)
        for row_idx, (emp_class, times) in enumerate(data.items(), start=1):
            color_rect = pink_color if (row_idx % 2) == 0 else dpink_color
            y_position = 600 + row_idx * row_height
            folding_time = times["folding_time"]
            folding_percentages = percentages[emp_class]["%f"]

            cvzone.putTextRect(img, emp_class, (-160 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

            cvzone.putTextRect(img, format_time(folding_time), (x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)
            cvzone.putTextRect(img, f"{(folding_percentages):.0f}%", (130 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

            cvzone.putTextRect(img, format_time(times["idle_time"]), (210 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)
            cvzone.putTextRect(img, f"{percentages[emp_class]['%i']:.0f}%", (340 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

            cvzone.putTextRect(img, format_time(times["offsite_time"]), (420 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)
            cvzone.putTextRect(img, f"{percentages[emp_class]['%o']:.0f}%", (550 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)


if __name__ == "__main__":
    emp_classes = ["Siti Umi", "Nina"]
    act_classes = ["Idle", "Folding"]
    video_path = "D:/AI_CCTV/.runs/videos/0624.mp4"
    mask_path = ".runs/images/mask6.png"
    emp_model_path = ".runs/detect/two_women/weights/best.pt"
    act_model_path = ".runs/detect/emp_gm1_rev/weights/best.pt"

    report_cctv = REPORT(video_path, mask_path, emp_model_path, act_model_path, emp_classes, act_classes)
    report_cctv()
