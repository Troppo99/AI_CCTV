import cv2
import os


def convert_video_to_frames(video_path, output_folder, percentage):
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Buka video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Tidak bisa membuka video.")
        return

    # Dapatkan jumlah frame dan frame rate dari video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Hitung jumlah frame yang akan diambil berdasarkan persentase
    num_frames_to_capture = int(total_frames * (percentage / 100.0))

    # Hitung interval pengambilan frame
    frame_interval = total_frames // num_frames_to_capture

    print(f"Total Frames: {total_frames}")
    print(f"Frames to Capture: {num_frames_to_capture}")
    print(f"Frame Interval: {frame_interval}")

    frame_count = 0
    captured_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Simpan frame berdasarkan interval yang dihitung
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_folder, f"frame_{captured_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            captured_count += 1

        frame_count += 1

        # Berhenti jika sudah mencapai jumlah frame yang diinginkan
        if captured_count >= num_frames_to_capture:
            break

    cap.release()
    print("Proses konversi selesai.")


# Contoh penggunaan
video_path = "../MY_FILES/Videos/CCTV/Train/10_ch04_20240425073845.mp4"
output_folder = "../MY_FILES/Videos/FRAME/video_10"
percentage = 10  # Misalnya 10% dari total durasi video

convert_video_to_frames(video_path, output_folder, percentage)
