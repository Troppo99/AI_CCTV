from ultralytics import YOLO
import cv2
import cvzone
import math


def result_elaboration(results, class_names):
    detection_list = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = class_names[cls]
            detection_list.append((x1, y1, x2, y2, conf, current_class))
    return detection_list


def main(video_path, model_act_path, model_track_path, scale, class_act, class_track):
    cap = cv2.VideoCapture(video_path)
    model_act = YOLO(model_act_path)
    model_track = YOLO(model_track_path)
    newDim = (int(1280 * scale), int(720 * scale))

    while True:
        ret, frame = cap.read()
        frame_region = cv2.bitwise_and(frame, cv2.imread(".runs/images/mask4.png"))
        results_act = model_act(frame_region, stream=True)
        results_track = model_track(frame_region, stream=True)
        detections_act = result_elaboration(results_act, class_act)
        detections_track = result_elaboration(results_track, class_track)

        for x1, y1, x2, y2, conf_track, current_class_track in detections_track:
            if current_class_track == "person" and conf_track > 0.25:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                for ax1, ay1, _, _, conf_act, current_class_act in detections_act:
                    if conf_act > 0.25:
                        cvzone.putTextRect(frame, f"{current_class_act} {conf_act}", (ax1, ay1), scale=2, thickness=2, colorT=(0, 0, 255), colorR=(0, 255, 255), colorB=(0, 252, 0), offset=5)

        frame_resized = cv2.resize(frame, newDim)
        cv2.imshow("Folding Area", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "D:/NWR27/AI_CCTV/.runs/videos/Folding Area Majalengka.mp4"
    model_act_path = ".runs/detect/act_gm1/weights/best.pt"
    model_track_path = ".runs/weights/yolov8l.pt"
    scale = 0.75
    class_act = ["Folding", "Iddle"]
    class_track = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    main(video_path, model_act_path, model_track_path, scale, class_act, class_track)
