from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture('Learn02/Videos/Murtaza/cars.mp4')

model = YOLO('Learn02/Yolo-Weights/yolov8n.pt') # Use yolov8l.pt untuk lebih bagus tapi lag parah jika tanpa GPU 

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread('Learn02/Images/mask.png')

while True:
    succes, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass=='car' or currentClass=='bus' or currentClass=='truck' or currentClass=='motorbike' and conf > 0.5:
                cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),
                                   scale = 1.5,thickness=2,colorT=(25,50,100),colorR=(0,255,255))
                cvzone.cornerRect(img, (x1,y1,w,h),l=9,t=3)

    cv2.imshow("Image", img)
    cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(0)