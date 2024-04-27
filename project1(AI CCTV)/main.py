import cv2
# cap = cv2.VideoCapture(0) #for default camera
cap = cv2.VideoCapture(0) #for web camera

while True:
    bo,frame = cap.read()
    cv2.imshow('Camera', frame)
    print(cap.isOpened())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("End...")