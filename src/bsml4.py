from ultralytics import YOLO
import cv2


def main(video_path, scale):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(".runs/weights/act.pt")

    while True:
        ret, frame = cap.read()
        results = model.track(source=frame, persist=True)
        annotated_frame = results[0].plot()

        # resize frame
        width = int(annotated_frame.shape[1] * scale)
        height = int(annotated_frame.shape[0] * scale)
        resized_frame = cv2.resize(annotated_frame, (width, height))
        cv2.imshow("Area Folding", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1"
    video_path = ".runs/videos/folding/01.mp4"
    main(video_path, scale=0.5)
