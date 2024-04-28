'''BASIC'''
# from ultralytics import YOLO
# import cv2

# # model = YOLO('Learn02/Yolo-Weights/yolov8n.pt')
# model = YOLO('Learn02/Yolo-Weights/yolov8l.pt')

# results = model('Learn02/Images/1.jpg',show=True)
# cv2.waitKey(0)

'''WEBCAM1'''
# from ultralytics import YOLO
# import cv2

# model = YOLO('Learn02/Yolo-Weights/yolov8n.pt')
# cap = cv2.VideoCapture(1)

# while True:
#     ret, frame = cap.read()
#     results = model(frame,show=True)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()

'''WEBCAM2'''
from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(1)
cap.set(3,1366)
cap.set(4,768)

model = YOLO('Learn02/Yolo-Weights/yolov8n.pt')

while True:
    succes, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)