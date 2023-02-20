import torch
import cv2

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s",pretrained=True)
model_1= torch.hub.load('ultralytics/yolov5', 'custom', path='D:\jeyaram\ML\simple_yolov5\\best.pt', force_reload=True)
video=cv2.VideoCapture(0)
while True:
    _,frame=video.read()
    results=model_1(frame)
    print(results)
    print(results.pandas().xyxy[0])
    x=results.pandas().xyxy[0]["xmin"].tolist()
    y=results.pandas().xyxy[0]["ymin"].tolist()
    X=results.pandas().xyxy[0]["xmax"].tolist()
    Y=results.pandas().xyxy[0]["ymax"].tolist()
    labels=results.pandas().xyxy[0]["name"].tolist()
    print(x,y,X,Y)
    for x1,y1,x2,y2,label in zip(x,y,X,Y,labels):
        print(x1,y1,x2,y2)
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
        cv2.putText(frame,str(label),(int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    if k == ord("q"):
        break


