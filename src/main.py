from bsml4 import AICCTV, REPORT, VideoSaver
import cv2


def where_server_sql(server):
    if server == "10.5.0.2":
        host = ""
        user = "robot"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3307
    elif server == "10.5.0.3":
        host = "localhost"
        user = "root"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3306
    return host, user, password, database, port


def main(
    emp_model_path=None,
    act_model_path=None,
    emp_classes=None,
    act_classes=None,
    video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
    anto_time=10,
    mask_path=None,
    saver=False,
    send=False,
    interval_send=10,
    table_sql="cam00",
    server=None,
):
    if server:
        send = True
        host, user, password, database, port = where_server_sql(server)
    ai_cctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path)
    report = REPORT(emp_classes, anto_time, interval_send)
    frame_rate = ai_cctv.cap.get(cv2.CAP_PROP_FPS)
    if saver:
        _, frame = ai_cctv.cap.read()
        base_path = ".runs/videos/writer"
        base_name = "monday"
        extension = ".mp4"
        file_name = VideoSaver.uniquifying(base_path, base_name, extension)
        video_saver = VideoSaver(file_name, frame.shape[1], frame.shape[0], frame_rate)
    mask = cv2.imread(mask_path) if mask_path is not None else None
    while ai_cctv.cap.isOpened():
        _, frame = ai_cctv.cap.read()
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) if mask_path is not None else None
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

        percentages = report.calculate_percentages()
        report.draw_table(frame, percentages)
        if saver:
            video_saver.write_frame(frame)
        frame = ai_cctv.resize_frame(frame)
        text_info = [
            f"Toleransi: {anto_time} detik",
            f"Masking: {mask_path}",
            f"Saver: {saver}",
            f"SQL: {send} to {server}",
        ]
        cv2.putText(frame, text_info[0], (1100, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, text_info[1], (1100, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, text_info[2], (1100, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, text_info[3], (1100, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow(f"Folding Area", frame)
        if send:
            report.send_to_sql(host, user, password, database, port, table_sql)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    if saver:
        video_saver.release()
    ai_cctv.cap.release()
    cv2.destroyAllWindows()
