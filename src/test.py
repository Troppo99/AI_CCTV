import torch
import cv2
from ultralytics import YOLO
import math
import cvzone


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
        frame_region = cv2.bitwise_and(frame, self.mask)

        # Process employee detection
        results_emp = self.model_emp(source=frame_region, stream=True)
        frame, emp_boxes_info = self.process_results(frame, results_emp, self.class_emp, (0, 255, 0))

        # Process activity detection
        results_act = self.model_act(source=frame_region, stream=True)
        frame, act_boxes_info = self.process_results(frame, results_act, self.class_act, (255, 0, 0))

        return frame, emp_boxes_info + act_boxes_info

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
    def draw_box(frame, x1, y1, x2, y2, class_id, conf, color, thickness=3, font_scale=3, font_thickness=3):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cvzone.putTextRect(frame, f"{class_id} {conf}", (max(0, x1), max(35, y1)), scale=font_scale, thickness=font_thickness)

    def __call__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, boxes_info = self.process_frame(frame)
            for box_info in boxes_info:
                self.draw_box(frame, *box_info)
            frame = self.resize_frame(frame)
            cv2.imshow("AI on Folding Area", frame)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    emp_classes = ["Umi", "Nina"]
    act_classes = ["Idle", "Folding"]
    video_path = "D:/AI_CCTV/.runs/videos/0624.mp4"
    mask_path = ".runs/images/mask6.png"
    emp_model_path = ".runs/detect/two_women/weights/best.pt"
    act_model_path = ".runs/detect/emp_gm1_rev/weights/best.pt"

    ai_cctv_processor = AICCTV(video_path, mask_path, emp_model_path, act_model_path, emp_classes, act_classes)
    ai_cctv_processor()
