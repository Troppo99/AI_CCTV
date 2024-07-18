import cv2

video_path = "D:/AI_CCTV/.runs/videos/test.mp4"
logo_path = "D:/AI_CCTV/.runs/images/graphic.png"

logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

logo_height, logo_width = logo.shape[:2]

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    _, frame = cap.read()

    frame_height, frame_width = frame.shape[:2]
    x_offset = frame_width - logo_width
    y_offset = frame_height - logo_height

    if logo.shape[2] == 4:
        logo_bgr = logo[:, :, :3]
        alpha_channel = logo[:, :, 3]
        mask = alpha_channel / 255.0
        inverse_mask = 1.0 - mask
        for c in range(0, 3):
            frame[y_offset : y_offset + logo_height, x_offset : x_offset + logo_width, c] = mask * logo_bgr[:, :, c] + inverse_mask * frame[y_offset : y_offset + logo_height, x_offset : x_offset + logo_width, c]
    else:
        frame[y_offset : y_offset + logo_height, x_offset : x_offset + logo_width] = logo

    height = int(frame.shape[0] * 0.4)
    width = int(frame.shape[1] * 0.4)
    frame_resized = cv2.resize(frame, (width, height))
    cv2.imshow("Video with Logo", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
