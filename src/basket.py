from ultralytics import YOLO
import cv2
import cvzone
import math
import time


def video_writer(cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / original_fps)

    return frame_width, frame_height, original_fps, frame_delay


def main(video_path, model_path, scale):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    classNames = ["0", "2", "3", "basket"]
    newDim = (int(1280 * scale), int(720 * scale))
    frame_width, frame_height, original_fps, frame_delay = video_writer(cap)
    out = cv2.VideoWriter(".runs/videos/test.avi", cv2.VideoWriter_fourcc(*"XVID"), original_fps, (frame_width, frame_height))

    while True:
        start_time = time.time()
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if conf > 0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cvzone.putTextRect(img, f"{currentClass} {conf}", (max(0, x1), max(35, y1)), scale=2, thickness=2, colorT=(0, 0, 255), colorR=(0, 255, 255), colorB=(0, 252, 0), offset=5)
        out.write(img)
        img_resized = cv2.resize(img, newDim)
        cv2.imshow("Image", img_resized)
        processing_time = time.time() - start_time
        wait_time = max(1, frame_delay - int(processing_time * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord("n"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "D:/NWR27/AI_CCTV/.runs/videos/record1_e.mp4"
    model_path = ".runs/detect/basket/weights/best.pt"
    scale = 0.75
    main(video_path, model_path, scale)
