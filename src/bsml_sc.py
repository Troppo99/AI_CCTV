from ultralytics import YOLO
import cv2
import cvzone
import math


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    return cap


def process_detections(results, class_names, conf_th):
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = class_names[cls]
            if conf > conf_th:
                detections.append((x1, y1, x2, y2, currentClass, conf))
    return detections


def main(video_path, model_emp_path, model_act_path, emp_conf_th, act_conf_th, screen_width, screen_height):
    cap = initialize_video_capture(video_path)

    model_emp = YOLO(model_emp_path)
    model_act = YOLO(model_act_path)

    class_names_emp = ["Neneng", "Imas", "Euis", "Siti", "Enok", "Puti", "Sausan", "Eti", "Atik", "Imam"]
    class_names_act = ["Wrapping", "unloading", "packing", "sorting"]

    while True:
        success, img = cap.read()
        if not success:
            break

        results_emp = model_emp(img, stream=True)
        results_act = model_act(img, stream=True)

        detections_emp = process_detections(results_emp, class_names_emp, emp_conf_th)
        detections_act = process_detections(results_act, class_names_act, act_conf_th)

        # Combine detections and display them
        for x1, y1, x2, y2, emp_class, emp_conf in detections_emp:
            act_detected = False
            for (
                ax1,
                ay1,
                ax2,
                ay2,
                act_class,
                act_conf,
            ) in detections_act:
                if (x1 <= ax1 <= x2 and y1 <= ay1 <= y2) or (x1 <= ax2 <= x2 and y1 <= ay2 <= y2):
                    act_detected = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cvzone.putTextRect(
                        img,
                        f"{emp_class} = {act_class}",
                        (max(0, x1), max(35, y1)),
                        scale=2,
                        thickness=2,
                        colorT=(0, 0, 255),
                        colorR=(0, 255, 255),
                        colorB=(0, 252, 0),
                        offset=5,
                    )
                    break
            if not act_detected:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cvzone.putTextRect(
                    img,
                    f"{emp_class} is idle",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 0),
                    colorR=(255, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )

        # Resize image to fit screen while maintaining aspect ratio
        height, width, _ = img.shape
        scale = min(screen_width / width, screen_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img, (new_width, new_height))

        # Create a blank image with the size of the screen
        screen_img = cv2.resize(resized_img, (screen_width, screen_height))
        screen_img.fill(0)

        # Calculate position to center the resized image
        x_offset = (screen_width - new_width) // 2
        y_offset = (screen_height - new_height) // 2

        # Place the resized image on the center of the blank image
        screen_img[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized_img

        cv2.imshow("Report Display AI CCTV", screen_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "../MY_FILES/Videos/CCTV/source/10_ch04_20240425073845.mp4"
    model_emp_path = ".runs/detect/.arc/employees-1/weights/best.pt"
    model_act_path = ".runs/detect/.arc/eactivity-1/weights/best.pt"
    emp_conf_th, act_conf_th = (0.8, 0.25)

    # Replace these with your screen resolution
    screen_width, screen_height = int(round(1920 * 0.5, 2)), int(round(1080 * 0.5, 2))

    main(video_path, model_emp_path, model_act_path, emp_conf_th, act_conf_th, screen_width, screen_height)
