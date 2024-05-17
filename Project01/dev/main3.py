import cv2

def main():
    # cap = cv2.VideoCapture("C:/NWR27/Folding Area_proses sortir.mp4")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Unable to access camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print("**********************End****************************")
