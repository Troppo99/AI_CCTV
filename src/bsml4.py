from ultralytics import YOLO
import cv2
import numpy as np
import cvzone


def main(video_path, scale):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(".runs/weights/act.pt")
    pink = (255, 0, 255)
    pink_reduce = (255, 200, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = cv2.imread(".runs/images/mask4.png")
        if mask is None:
            raise ValueError("Mask image not found")
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        frame_region = cv2.bitwise_and(frame, mask_resized)

        results = model.track(source=frame_region, persist=True)

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    if box.id is not None and box.id[0] is not None:
                        x1, y1, x2, y2 = box.xyxy[0]  # Convert to CPU and then to NumPy
                        track_id = int(box.id[0])  # Get the track ID
                        cls = int(box.cls[0])  # Get the class ID
                        class_name = model.names[cls]  # Get the class name

                        # Draw bounding box
                        warna = pink if class_name == "Folding" else pink_reduce
                        cvzone.putTextRect(frame, f"{class_name} ID: {track_id}", (int(x1), int(y1 - 10)), scale=1, thickness=1, colorR=warna)
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
    # video_path = ".runs/videos/folding/01.mp4"
    main(video_path, scale=0.75)
