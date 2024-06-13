import cv2
import time

# Path ke file video
video_path = ".runs/videos/mouse.mp4"

# Waktu-waktu yang diinginkan untuk menyimpan frame (dalam detik)
save_times = [5, 10, 15]  # Menyimpan frame pada detik ke-5, 10, dan 15

# Membuka video
cap = cv2.VideoCapture(video_path)

# Memastikan video berhasil dibuka
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Mengambil FPS (Frame Per Second) dari video
fps = cap.get(cv2.CAP_PROP_FPS)

# Menghitung frame yang diinginkan berdasarkan waktu
save_frames = [int(time * fps) for time in save_times]

frame_counter = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if frame_counter in save_frames:
        # Simpan frame ke file
        frame_time = save_times[save_frames.index(frame_counter)]
        filename = f"FRAME/frame_at_{frame_time}_seconds.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved at {frame_time} seconds as {filename}")

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
