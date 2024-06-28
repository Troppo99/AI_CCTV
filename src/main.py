from bsml4 import AICCTV, REPORT
import cv2


def main(emp_model_path, act_model_path, emp_classes, act_classes, video_path, mask_path=None):
    ai_cctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path)
    report = REPORT(emp_classes)
    """ #######-> Start of Video Saver 1 <-####### """
    # ret, frame = ai_cctv.cap.read()
    # video_saver = VideoSaver(".runs/videos/writer/output_video.mp4", frame.shape[1], frame.shape[0], frame_rate)
    """ --------> End of Video Saver 1 <-------- """
    """ #######-> Start of overlay 2 <-####### """
    # table_bg = cv2.imread(".runs/images/OL1.png", cv2.IMREAD_UNCHANGED)
    # new_width = 1350
    # aspect_ratio = table_bg.shape[1] / table_bg.shape[0]
    # new_height = int(new_width / aspect_ratio)
    # table_bg = cv2.resize(table_bg, (new_width, new_height))
    """ --------> End of overlay 2 <-------- """
    frame_rate = ai_cctv.cap.get(cv2.CAP_PROP_FPS)
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

        """ #######-> Start of overlay 2 <-####### """
        # frame = cvzone.overlayPNG(frame, table_bg, (1800, 1055))
        # cv2.imshow("Video with Overlay", frame_with_overlay)
        """ --------> End of overlay 2 <-------- """
        percentages = report.calculate_percentages()
        report.draw_table(frame, percentages)

        """ #######-> Start of Video Saver 1 <-####### """
        # video_saver.write_frame(frame)
        """ --------> End of Video Saver 1 <-------- """

        frame = ai_cctv.resize_frame(frame)
        cv2.imshow(act_model_path, frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    """ #######-> Start of Video Saver 1 <-####### """
    # video_saver.release()
    """ --------> End of Video Saver 1 <-------- """
    ai_cctv.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("This is main!\n")
