'''BASIC'''
# from ultralytics import YOLO
# import cv2

# # model = YOLO('Learn02/Yolo-Weights/yolov8n.pt')
# model = YOLO('Learn02/Yolo-Weights/yolov8l.pt')

# # results = model('Learn02/Images/1.jpg',show=True)
# results = model('project1(AI CCTV)\Data set\Folding Area_proses sortir.mp4',show=True)
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
import math

cap = cv2.VideoCapture(1)
cap.set(3,1080)
cap.set(4,720)

model = YOLO('Learn02/Yolo-Weights/yolov8n.pt')

while True:
    succes, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.reangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            # print(x1,y1,x2,y2)
            
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img, (x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(y1-20)))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()