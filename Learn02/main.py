# Looping
while True:
    success, img = cap.read()
    if not success:  # Check if the video has ended or if there's a reading error
        break

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass in ["car", "bus", "truck", "motorbike"]) and conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    for result in resultTracker:
        x1, y1, x2, y2, Id = map(int, result)
        cvzone.putTextRect(
            img,
            f"{classNames[int(Id)]} {conf:.2f} ID:{Id}",
            (max(0, x1), max(35, y1)),
            scale=1.5,
            thickness=2,
            colorT=(25, 50, 100),
            colorR=(0, 255, 255),
        )
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, t=3)

    img = cv2.resize(img, newDim)
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
