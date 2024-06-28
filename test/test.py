import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO


class ObjectDetections:

    def __init__(
        self,
        capture_index="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
        model_path=".runs/weights/yolov8l.pt",
        is_saved=False,
    ):
        self.capture_index = capture_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)
        self.model_path = model_path
        self.model = self.load_model(model_path)
        self.is_saved = is_saved

    def load_model(self, model_path):
        model = YOLO(model_path)
        model.fuse()
        return model

    def predict(self, frame, is_saved):
        results = self.model(frame, save=is_saved)
        return results

    def plot_bboxes(self, results):
        xyxys = []
        confidences = []
        class_ids = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if len(box) == 4:
                    xyxys.append(box[:4])
                    confidences.append(box[4])
                    class_ids.append(int(box[5]))

        return results[0].plot(), xyxys, confidences, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time()
            results = self.predict(frame, self.is_saved)
            annotated_frame, xyxys, confidences, class_ids = self.plot_bboxes(results)
            end_time = time()
            fps = 1 / (end_time - start_time)

            for i, box in enumerate(xyxys):
                if len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    confidence = confidences[i]
                    class_id = class_ids[i]
                    label = f"{class_id} {confidence:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            """ #######-> Start of experiment [1] <-####### """
            scale = 0.25  # 0 to 1
            annotated_frame = cv2.resize(annotated_frame, (int(cv2.CAP_PROP_FRAME_HEIGHT * scale * 1000), int(cv2.CAP_PROP_FRAME_WIDTH * scale * 1000)))
            """ --------> End of experiment [1] <-------- """

            cv2.imshow(self.model_path, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        cap.release()
        cv2.destroyAllWindows()


detection = ObjectDetections(
    # capture_index="D:/AI_CCTV/.runs/videos/0624.mp4",
    model_path=".runs/detect/person/weights/best.pt",
    # is_saved=True,
)
detection()
