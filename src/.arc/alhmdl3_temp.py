import cv2
from ultralytics import YOLO
import time
from datetime import timedelta

# Inisialisasi model YOLOv8
model = YOLO(".runs/weights/yolov8l.pt")

# Buka file video
video_path = ".runs/videos/area_line_tengah.mp4"
cap = cv2.VideoCapture(video_path)

# Inisialisasi dictionary untuk menyimpan waktu deteksi per ID
detection_times = {}


def format_time(waktu):
    return str(timedelta(seconds=int(waktu)))


# Loop melalui frame video
while cap.isOpened():
    success, frame = cap.read()
    frame_region=cv2.bitwise_and(frame, cv2.imread(".runs/images/mask2.png"))
    if success:
        results = model.track(frame_region, persist=True)
        annotated_frame = results[0].plot()

        # Proses hasil deteksi
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if hasattr(box, "id"):
                        track_id = int(box.id[0])
                        current_time = time.time()

                        if track_id not in detection_times:
                            detection_times[track_id] = current_time
                        else:
                            detection_duration = current_time - detection_times[track_id]
                            cv2.putText(annotated_frame, f"ID {track_id} Time: {format_time(detection_duration)}", (int(box.xyxy[0][0]), int(box.xyxy[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tampilkan frame yang telah dianotasi
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# Cetak durasi deteksi per ID setelah video selesai
for track_id, start_time in detection_times.items():
    detection_duration = time.time() - start_time
    print(f"ID {track_id} detected for {format_time(detection_duration)}")
