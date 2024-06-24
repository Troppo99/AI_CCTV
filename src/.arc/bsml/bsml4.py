from ultralytics import YOLO
import cv2
import numpy as np
import cvzone


def main(video_path, scale):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(".runs/weights/yolov8l.pt")
    pink = (255, 0, 255)
    pink_reduce = (255, 200, 255)

    while True:
        _, frame = cap.read()
        results = model.track(source=frame, persist=True)

        for r in results:
            if r.boxes:
                for box in r.boxes:
                    if box.id is not None and box.id[0] is not None:
                        x1, y1, x2, y2 = box.xyxy[0]
                        id = int(box.id[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        warna = pink if class_name == "person" else pink_reduce
                        cvzone.putTextRect(frame, f"{class_name} ID: {id}", (int(x1), int(y1 - 10)), scale=2, thickness=2, colorR=warna)
                    else:
                        print("Box ID is None, skipping")

        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        resized_frame = cv2.resize(frame, (width, height))
        cv2.imshow("Area Folding", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1"
    video_path = ".runs/videos/folding/01.mp4"
    main(video_path, scale=0.75)
