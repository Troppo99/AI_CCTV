from bsml4 import AICCTV, REPORT
import cv2
import queue
import concurrent.futures


def main(model_path, act_model_path, classes, act_classes, video_path, toogle=False, list_conf=[0, 0.2, 0.5, 0.8, 0.9], count=0, send=False, host=None, table="empact", data_loaded=True):
    aicctv = AICCTV(model_path, act_model_path, classes, act_classes, video_path, host)
    report = REPORT(aicctv.classes, data_loaded=data_loaded)

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
                mask = cv2.resize(cv2.imread(".runs/images/mask9.png"), (frame.shape[1], frame.shape[0]))

                """ USER CODE BEGIN: DECORATION --------------------------------- """
                cv2.putText(frame, "PRESENCE OF FOLDING EMPLOYEES PT GGI", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence Threshold: {list_conf[count]}", (1520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                """ USER CODE END: DECORATION ----------------------------------- """

                """ USER CODE BEGIN: RESULTS PROCESSING ------------------------- """
                frame, boxes_info, act_boxes_info = aicctv.process_frame(frame, list_conf[count], mask)
                for x1, y1, x2, y2, cls, conf, clr in boxes_info:
                    act_detected = False
                    for ax1, ay1, ax2, ay2, acls, aconf, aclr in act_boxes_info:
                        if aicctv.is_overlapping((x1, y1, x2, y2), (ax1, ay1, ax2, ay2)):
                            act_detected = True
                            report.update_data(cls, "folding", frame_duration)
                            aicctv.draw_label(frame, x1, y1, x2, y2, f"{cls} is {acls}", color=aclr)
                            break
                    if not act_detected:
                        report.update_data(cls, "idle", frame_duration)
                        aicctv.draw_label(frame, x1, y1, x2, y2, f"{cls} is idle", color=clr)

                detected_employees = [cls for _, _, _, _, cls, _, _ in boxes_info]
                for emp in aicctv.classes:
                    if emp not in detected_employees:
                        report.update_data(emp, "offsite", frame_duration)
                """ USER CODE END: RESULTS PROCESSING --------------------------- """

                report.draw_report(frame, toogle=toogle)
                frame_resized = aicctv.resize_frame(frame, 0.4)
                cv2.imshow("Folding Room", frame_resized)
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
    model_path="D:/AI_CCTV/.runs/detect/emp-m/weights/best.pt",
    act_model_path="D:/AI_CCTV/.runs/detect/fold-m/weights/best.pt",
    classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Umi", "Hutizah", "Anjani", "Tia"],
    act_classes=["Folding"],
    video_path="rtsp://admin:oracle2015@192.168.100.6:554/Streaming/Channels/1",
    data_loaded=False,
    # host="localhost",
    # host="10.5.0.2",
)
