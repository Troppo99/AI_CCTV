from bsml4 import AICCTV, REPORT, capture_frame, cv2, time
import concurrent.futures
import queue


def main(
    emp_model_path=".runs/detect/emp-m/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    emp_classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Umi", "Hutizah", "Anjani", "Tia"],
    act_classes=["Working"],
    video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
    anto_time=3,
    mask_path=".runs/images/mask8.png",
    send=False,
    table_sql="empact",
    server="10.5.0.3",
    camera_id="FOLDING",
    show=False,
):
    start_time = time.time()
    ai_cctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path, server)
    report = REPORT(emp_classes, anto_time)
    frame_rate = ai_cctv.cap.get(cv2.CAP_PROP_FPS)
    frame_queue = queue.Queue()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(capture_frame, ai_cctv.cap, frame_queue)
        if server:
            send = True
            host, user, password, database, port = report.where_sql_server(server)
        mask = cv2.imread(mask_path) if mask_path is not None else None
        while ai_cctv.cap.isOpened():
            if not frame_queue.empty():
                frame = frame_queue.get()
                if frame is None:
                    current_time = time.time()
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"- - -\nframe is None, Buddy! When it was {timestamp}")
                    print(f"Program is running for {current_time-start_time:.0f}!\n- - -")
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) if mask is not None else None
                frame_duration = 1 / frame_rate
                frame, emp_boxes_info, act_boxes_info = ai_cctv.process_frame(frame, mask_resized)
                for x1, y1, x2, y2, emp_class, _, emp_color in emp_boxes_info:
                    act_detected = False
                    for ax1, ay1, ax2, ay2, act_class, _, act_color in act_boxes_info:
                        if ai_cctv.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                            act_detected = True
                            report.update_data_table(emp_class, "working_time", frame_duration)
                            text = f"{emp_class} is {act_class}"
                            ai_cctv.draw_box(frame, x1, y1, x2, y2, text, act_color)
                            break
                    if not act_detected:
                        report.update_data_table(emp_class, "idle_time", frame_duration)
                        text = f"{emp_class} is idle"
                        ai_cctv.draw_box(frame, x1, y1, x2, y2, text, emp_color)
                detected_employees = [emp_class for _, _, _, _, emp_class, _, _ in emp_boxes_info]
                for emp_class in emp_classes:
                    if emp_class not in detected_employees:
                        report.update_data_table(emp_class, "offsite_time", frame_duration)
                if show:
                    percentages = report.calculate_percentages()
                    report.draw_table(frame, percentages)
                frame = ai_cctv.resize_frame(frame)
                if show:
                    mask_info = mask_path.split("/")[-1] if mask_path else mask_path
                    data_info = f"Sending to {host}" if send else "Not sending"
                    text_info = [
                        f"Tolerance: {anto_time} seconds",
                        f"Masking: {mask_info}",
                        f"Data: {data_info}",
                    ]
                    j = len(text_info) if server else len(text_info) - 1
                    for i in range(j):
                        cv2.putText(frame, text_info[i], (980, 30 + i * 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(f"Folding Area", frame)
                if send:
                    report.send_to_sql(host, user, password, database, port, table_sql, camera_id)
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    break

    ai_cctv.cap.release()
    cv2.destroyAllWindows()


main(
    anto_time=300,
    show=True,
    # server="10.5.0.2",
)
