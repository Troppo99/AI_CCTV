"""
Program ini adalah program user,
yang akan berbeda tiap projek.
Program ini hanya untuk 'Folding Area'.
@CopyRight2024, TroppoLungo
"""

# impor library
from bsml4 import AICCTV, REPORT, SAVER, capture_frame, cv2, time
import concurrent.futures
import queue


# fungsi di file main
def main(
    emp_model_path=".runs/detect/emp-m/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    emp_classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Umi", "Hutizah", "Anjani", "Tia"],
    act_classes=["Working"],
    video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
    anto_time=3,
    mask_path=None,
    save=False,
    send=False,
    interval_send=1,
    table_sql="empact",
    server=None,
    camera_id="FOLDING",
    show=False,
):
    # memulai waktu di program
    start_time = time.time()

    # inisialisasi instances dari library bsml4
    ai_cctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path)
    report = REPORT(emp_classes, anto_time, interval_send)

    # mengambil data fps
    frame_rate = ai_cctv.cap.get(cv2.CAP_PROP_FPS)

    # menyimpan frame ke dalam queue atau antrian
    frame_queue = queue.Queue()

    # proses mengambil frame yang akan dimasukan kedalam queue secara multiprocessing
    with concurrent.futures.ThreadPoolExecutor() as executor:

        # eksekusi pengumpulan frame ke dalam queue
        executor.submit(capture_frame, ai_cctv.cap, frame_queue)

        # apakah data dikirim ke mySQL
        if server:
            send = True
            host, user, password, database, port = report.where_sql_server(server)

        # apakah video direkam
        if save:
            _, frame = ai_cctv.cap.read()
            base_path = ".runs/videos/writer"
            base_name = "exp"
            extension = ".mp4"
            file_name = SAVER.uniquifying(base_path, base_name, extension)
            video_saver = SAVER(file_name, frame.shape[1], frame.shape[0], frame_rate)

        # memuat masking gambar
        mask = cv2.imread(mask_path) if mask_path is not None else None

        # looping
        while ai_cctv.cap.isOpened():

            # apakah queue tidak kosong
            if not frame_queue.empty():

                # proses penyimpanan frame ke dalam queue
                frame = frame_queue.get()

                # pendeteksian apakah frame 'Nonetype'
                if frame is None:
                    current_time = time.time()
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"- - -\nframe is None, Buddy! When it was {timestamp}")
                    print(f"Program is running for {current_time-start_time:.0f}!\n- - -")

                # penyesuaian ukuran masking
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) if mask is not None else None

                # mengambil data kecepatan frame dari sumber video
                frame_duration = 1 / frame_rate

                # pengambilan data objek dari tiap frame yang looping
                frame, emp_boxes_info, act_boxes_info = ai_cctv.process_frame(frame, mask_resized)

                # setiap data objek 'emp' dilacak
                for x1, y1, x2, y2, emp_class, _, emp_color in emp_boxes_info:

                    # activity dinyatakan tidak terdeteksi
                    act_detected = False

                    # saat activity terdeteksi, data objek 'act' dilacak
                    for ax1, ay1, ax2, ay2, act_class, _, act_color in act_boxes_info:

                        # apakah saat 'emp' terdeteksi 'act' terdeteksi
                        if ai_cctv.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):

                            # activity dinyatakan terdeteksi
                            act_detected = True

                            # menyimpan data waktu tiap acitivity per employee
                            report.update_data_table(emp_class, "working_time", frame_duration)
                            text = f"{emp_class} is {act_class}"
                            ai_cctv.draw_box(frame, x1, y1, x2, y2, text, act_color)
                            break

                    # apakah 'act' tidak terdeteksi
                    if not act_detected:

                        # menyimpan data waktu 'idle' per employee
                        report.update_data_table(emp_class, "idle_time", frame_duration)
                        text = f"{emp_class} is idle"
                        ai_cctv.draw_box(frame, x1, y1, x2, y2, text, emp_color)

                # menyusun 'emp' yang terbaca
                detected_employees = [emp_class for _, _, _, _, emp_class, _, _ in emp_boxes_info]

                # melacak 'emp' yang terdeteksi
                for emp_class in emp_classes:

                    # apakah ada employee dari list kelas asal tapi tidak terlacak di 'emp' yang terdeteksi
                    if emp_class not in detected_employees:

                        # menympan 'emp' tersebut ke data sebagai 'offsite'
                        report.update_data_table(emp_class, "offsite_time", frame_duration)

                # apakah ingin menampilkan display
                if show:
                    percentages = report.calculate_percentages()
                    report.draw_table(frame, percentages)

                # apakah display sedang direkam
                if save:
                    video_saver.write_frame(frame)

                # apakah ingin menampilkan display
                if show:

                    # mengurangi ukuran frame sebelum ditampilkan (karena size kamera dan monitor beda)
                    frame = ai_cctv.resize_frame(frame)

                    # menampilkan informasi sistem program saat didisplaykan (bukan data waktu 'emp' inimah)
                    mask_info = mask_path.split("/")[-1] if mask_path else mask_path
                    saver_info = "Recording" if save else "Not Recording"
                    data_info = f"Sending to {host}" if send else "Not sending"
                    text_info = [
                        f"Tolerance: {anto_time} seconds",
                        f"Masking: {mask_info}",
                        f"Saver: {saver_info}",
                        f"Data: {data_info}",
                        f"Interval Send: {interval_send} seconds",
                    ]
                    j = len(text_info) if server else len(text_info) - 1
                    for i in range(j):
                        cv2.putText(frame, text_info[i], (980, 30 + i * 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(f"Folding Area", frame)

                # apakah data waktu dikirim ke mySQL server
                if send:
                    report.send_to_sql(host, user, password, database, port, table_sql, camera_id)

                # menunggu keyboard ditekan selama 1ms (nanti akan mengeluarkan biner dari ASCII yang ditekan)
                # dan mengoperasikan dengan AND 0xFF (hex: karena ouput waitkey adalah 8bit) (ascii AND 11111111)
                # (misal 'n' ASCII = 110 dan binernya 01101110) maka bitwise AND nya adalah 110
                # kemudian dibandingkan hasilnya apakah sama dengan biner kode ASCII 'n' (110)
                if cv2.waitKey(1) & 0xFF == ord("n"):

                    # program diberhentikan (while loop utama diputuskan)
                    break

    # apakah display sedang direkam
    if save:
        video_saver.release()

    # putuskan semua sumber daya CV2
    ai_cctv.cap.release()
    cv2.destroyAllWindows()
