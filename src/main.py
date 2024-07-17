from bsml4 import AICCTV, REPORT
import cv2
import queue
import concurrent.futures


def main(model_path, classes, video_path, toogle=False, list_conf=[0, 0.2, 0.5, 0.8, 0.9], count=0, send=False, host=None, table="presence", data_loaded=True):
    aicctv = AICCTV(model_path, classes, video_path, host)
    report = REPORT(aicctv.classes, data_loaded=True)

    frame_queue = queue.Queue(maxsize=10)
    frame_rate = aicctv.cap.get(cv2.CAP_PROP_FPS)
    if host:
        send = True
        user, password, database, port = report.server_address(host)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(aicctv.capture_frame, frame_queue)
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_duration = 1 / frame_rate

                """ USER CODE BEGIN: DECORATION --------------------------------- """
                cv2.putText(frame, "PRESENCE OF ROBOTIC EMPLOYEES PT GGI", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence Threshold: {list_conf[count]}", (1520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                """ USER CODE END: DECORATION ----------------------------------- """

                """ USER CODE BEGIN: RESULTS PROCESSING ------------------------- """
                frame, boxes_info = aicctv.process_frame(frame, list_conf[count])
                detected_employees = [cls for _, _, _, _, cls, _ in boxes_info]
                for emp in aicctv.classes:
                    if emp in detected_employees:
                        report.update_data(emp, "onsite", frame_duration)
                    else:
                        report.update_data(emp, "offsite", frame_duration)
                for x1, y1, x2, y2, cls, conf in boxes_info:
                    aicctv.draw_label(frame, x1, y1, x2, y2, f"{cls} {conf}")
                report.draw_report(frame, toogle=toogle)
                """ USER CODE END: RESULTS PROCESSING --------------------------- """

                frame_resized = aicctv.resize_frame(frame, 0.6)
                cv2.imshow("Robotic Room", frame_resized)
                if send:
                    report.send_data(host, user, password, database, port, table)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    toogle = not toogle
                elif key == ord("c"):
                    count += 1
                    if count > 4:
                        count = 0
                elif key == ord("n"):
                    break
        aicctv.cap.release()
        cv2.destroyAllWindows()


main(
    model_path="D:/AHMDL/.runs/detect/robemp_v2l/weights/best.pt",
    classes=["Fathria", "Nana", "Nurdin", "Rizki", "Waskita"],
    video_path="rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1",
    data_loaded=False,
    # host="localhost",
    # host="10.5.0.2",
)
