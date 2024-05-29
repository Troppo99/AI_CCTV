from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize video capture
cap = cv2.VideoCapture("../MY_FILES/Videos/CCTV/New/Recording Video - 28 Mei.mp4")

# Initialize YOLO model
model = YOLO("D:/NWR27/AI_CCTV/runs/detect/anyar/weights/last.pt")

# Class names
classNames = ["Sorting", "Wrapping", "Packaging", "Idle"]

# Dimensions for imshow
scaleof = 0.75  # 0 to 1.5 (1280 x 720 default video resolution)
newDim = (int(1280 * scaleof), int(720 * scaleof))

# Get frame width, height, and original frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / original_fps)  # Delay in milliseconds

# Initialize VideoWriter object with the original frame rate
out = cv2.VideoWriter(
    "runs/videos/output_video.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    original_fps,
    (frame_width, frame_height),
)

while True:
    start_time = time.time()
    success, img = cap.read()
    if not success:
        print("...Gagal muka cv2 bro...")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.25:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorT=(0, 0, 255),
                    colorR=(0, 255, 255),
                    colorB=(0, 252, 0),
                    offset=5,
                )

    # Write the frame to the video file
    out.write(img)

    # Display the frame
    img_resized = cv2.resize(img, newDim)
    cv2.imshow("Image", img_resized)

    # Calculate processing time and add delay
    processing_time = time.time() - start_time
    wait_time = max(
        1, frame_delay - int(processing_time * 1000)
    )  # Ensure non-negative wait time
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()
