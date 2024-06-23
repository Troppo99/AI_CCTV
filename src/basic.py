import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO


class ObjectDetections:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)
        self.model = self.load_model()

    def load_model(self):
        model = YOLO("yolov8n.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results):
        xyxys = []
        confidences = []
        class_ids = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        return results[0].plot(), xyxys, confidences, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            assert ret
            start_time = time()

            results = self.predict(frame)
            annotated_frame, xyxys, confidences, class_ids = self.plot_bboxes(results)

            end_time = time()
            fps = 1 / (end_time - start_time)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Detection", annotated_frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Esc key to stop
                break

        cap.release()
        cv2.destroyAllWindows()


# Contoh penggunaan
if __name__ == "__main__":
    detection = ObjectDetections(capture_index=0)
    detection()
