from bsml4 import AICCTV, REPORT
import cv2
import queue
import concurrent.futures


def main(emp_model_path, act_model_path, emp_classes, act_classes, video_path, send=False, host=None, table="empact", mask_path=None, show=False, load_data=True):
    aicctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path, host)
    report = REPORT(aicctv.emp_classes, load_data=load_data)

    frame_queue = queue.Queue(maxsize=10)
    frame_rate = aicctv.cap.get(cv2.CAP_PROP_FPS)
    mask = cv2.imread(mask_path) if mask_path is not None else None
    if host:
        send = True
        user, password, database, port = report.server_address(host)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(aicctv.capture_frame, frame_queue)
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_duration = 1 / frame_rate
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) if mask is not None else None

                """ USER CODE BEGIN: RESULTS PROCESSING ------------------------- """
                frame, emp_boxes_info, act_boxes_info = aicctv.process_frame(frame, mask_resized)
                for x1, y1, x2, y2, emp, _, emp_color in emp_boxes_info:
                    act_detected = False
                    for ax1, ay1, ax2, ay2, act_class, _, act_color in act_boxes_info:
                        if aicctv.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                            act_detected = True
                            report.update_data(emp, "working_time", frame_duration)
                            text = f"{emp} is {act_class}"
                            aicctv.draw_label(frame, x1, y1, x2, y2, text, act_color)
                            break
                    if not act_detected:
                        report.update_data(emp, "idle_time", frame_duration)
                        text = f"{emp} is idle"
                        aicctv.draw_label(frame, x1, y1, x2, y2, text, emp_color)

                detected_employees = [emp for _, _, _, _, emp, _, _ in emp_boxes_info]
                for emp in emp_classes:
                    if emp not in detected_employees:
                        report.update_data(emp, "offsite_time", frame_duration)
                """ USER CODE END: RESULTS PROCESSING --------------------------- """

                if show:
                    percentages = report.calculate_percentages()
                    report.draw_report(frame, percentages)
                    frame = aicctv.resize_frame(frame)
                    mask_info = mask_path.split("/")[-1] if mask_path else mask_path
                    data_info = f"Sending to {host}" if send else "Not sending"
                    text_info = [f"Masking: {mask_info}", f"Data: {data_info}"]
                    j = len(text_info) if host else len(text_info) - 1
                    for i in range(j):
                        cv2.putText(frame, text_info[i], (980, 30 + i * 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(f"Folding", frame)
                if send:
                    report.send_data(host, user, password, database, port, table)
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    break

        aicctv.cap.release()
        cv2.destroyAllWindows()


main(
    emp_model_path=".runs/detect/emp-m/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    emp_classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Umi", "Hutizah", "Anjani", "Tia"],
    act_classes=["Working"],
    video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
    mask_path=".runs/images/mask8.png",
    # host="localhost",
    # host="10.5.0.2",
    show=True,
    load_data=False,
)
