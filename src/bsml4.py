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
        video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
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
                "folding_time": 0,
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

        cv2.putText(frame, f"Report Table", (-140 + x_move, 540 + y_move), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.8, (20, 200, 20), 2, cv2.LINE_AA)
        headers = ["Employee", "Folding", "Idle", "Offsite"]

        cv2.putText(frame, headers[0], (-160 + x_move, 595 + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, headers[1], (90 + x_move, 595 + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, headers[2], (430 + x_move, 595 + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, headers[3], (770 + x_move, 595 + y_move), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        for row_idx, (emp_class, times) in enumerate(self.data.items(), start=1):
            color_rect = pink_color if (row_idx % 2) == 0 else dpink_color
            y_position = 610 + row_idx * row_height

            cvzone.putTextRect(frame, emp_class, (-160 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

            cvzone.putTextRect(frame, format_time(times["folding_time"]), (90 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)
            cvzone.putTextRect(frame, f"{(percentages[emp_class]['%f']):.0f}%", (285 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

            cvzone.putTextRect(frame, format_time(times["idle_time"]), (430 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)
            cvzone.putTextRect(frame, f"{percentages[emp_class]['%i']:.0f}%", (625 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)

            cvzone.putTextRect(frame, format_time(times["offsite_time"]), (770 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)
            cvzone.putTextRect(frame, f"{percentages[emp_class]['%o']:.0f}%", (965 + x_move, y_position + y_move), scale=scale_text, thickness=2, offset=5, colorR=color_rect)


class VideoSaver:
    def __init__(self, output_path, frame_width, frame_height, fps=20.0):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()


def main(mask_path, emp_model_path, act_model_path, emp_classes, act_classes, video_path):
    # Create Instances
    ai_cctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path)
    report = REPORT(emp_classes)

    """ #######-> Start of Video Saver 1 <-####### """
    # ret, frame = ai_cctv.cap.read()
    # video_saver = VideoSaver(".runs/videos/writer/output_video.mp4", frame.shape[1], frame.shape[0], frame_rate)
    """ --------> End of Video Saver 1 <-------- """

    """ #######-> Start of overlay 2 <-####### """
    # table_bg = cv2.imread(".runs/images/OL1.png", cv2.IMREAD_UNCHANGED)
    # new_width = 1350
    # aspect_ratio = table_bg.shape[1] / table_bg.shape[0]
    # new_height = int(new_width / aspect_ratio)
    # table_bg = cv2.resize(table_bg, (new_width, new_height))
    """ --------> End of overlay 2 <-------- """

    frame_rate = ai_cctv.cap.get(cv2.CAP_PROP_FPS)
    mask = cv2.imread(mask_path)
    while ai_cctv.cap.isOpened():
        _, frame = ai_cctv.cap.read()
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        frame_duration = 1 / frame_rate

        frame, emp_boxes_info, act_boxes_info = ai_cctv.process_frame(frame, mask_resized)
        for x1, y1, x2, y2, emp_class, _, emp_color in emp_boxes_info:
            act_detected = False
            for ax1, ay1, ax2, ay2, act_class, _, act_color in act_boxes_info:
                if ai_cctv.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                    act_detected = True
                    report.update_data_table(emp_class, act_class.lower() + "_time", frame_duration)
                    text = f"{emp_class} is {act_class}"
                    ai_cctv.draw_box(frame, x1, y1, x2, y2, text, act_color)
                    break
            if not act_detected:
                report.update_data_table(emp_class, "idle_time", frame_duration)
                text = f"{emp_class} is idle"
                ai_cctv.draw_box(frame, x1, y1, x2, y2, text, emp_color)
        detected_employees = [emp_class for _, _, _, _, emp_class, _, _ in emp_boxes_info]
        for emp_class in emp_classes:
            if emp_class not in detected_employees:
                report.update_data_table(emp_class, "offsite_time", frame_duration)

        """ #######-> Start of overlay 2 <-####### """
        # frame = cvzone.overlayPNG(frame, table_bg, (1800, 1055))
        # cv2.imshow("Video with Overlay", frame_with_overlay)
        """ --------> End of overlay 2 <-------- """
        percentages = report.calculate_percentages()
        report.draw_table(frame, percentages)

        """ * * *---> Start of Video Saver 1 <---* * * """
        # video_saver.write_frame(frame)
        """ --------> End of Video Saver 1 <-------- """

        frame = ai_cctv.resize_frame(frame)
        cv2.imshow("AI on Folding Area", frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    """ * * *---> Start of Video Saver 1 <---* * * """
    # video_saver.release()
    """ --------> End of Video Saver 1 <-------- """

    ai_cctv.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hello Nana Wartana!\n")
