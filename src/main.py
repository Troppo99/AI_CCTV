from bsml4 import AICCTV, REPORT
import cv2
import concurrent.futures
import queue
import time


def main(emp_model_path, act_model_path, emp_classes, act_classes, video_path, send=False, host=None, table="empact", mask_path=None, show=False, load_data=True):
    aicctv = AICCTV(emp_model_path, act_model_path, emp_classes, act_classes, video_path, host)
    report = REPORT(aicctv.emp_classes, load_data=load_data)
    print("wait for 2 seconds...")
    time.sleep(2)
    print("wait for 1 seconds...")
    time.sleep(1)

    frame_queue = queue.Queue(maxsize=10)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(aicctv.capture_frame, frame_queue)
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()

                """ USER CODE BEGIN: DECORATION --------------------------------- """
                cv2.putText(frame, "PE OF FOLDING EMPLOYEES PT GGI", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                """ USER CODE END: DECORATION ----------------------------------- """

                frame = aicctv.resize_frame(frame, 0.3)
                cv2.imshow(f"Folding", frame)
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
