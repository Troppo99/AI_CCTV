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
            for box in boxes:
                if len(box) == 4:
                    xyxys.append(box[:4])
                    confidences.append(box[4])
                    class_ids.append(int(box[5]))

        return results[0].plot(), xyxys, confidences, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time()
            results = self.predict(frame)
            annotated_frame, xyxys, confidences, class_ids = self.plot_bboxes(results)
            end_time = time()
            fps = 1 / (end_time - start_time)

            # Plot the bounding boxes and labels on the frame
            for i, box in enumerate(xyxys):
                if len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    confidence = confidences[i]
                    class_id = class_ids[i]
                    label = f"{class_id} {confidence:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow("YOLOv8 Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        cap.release()
        cv2.destroyAllWindows()


# Contoh penggunaan
if __name__ == "__main__":
    detection = ObjectDetections(capture_index=0)
    detection()
