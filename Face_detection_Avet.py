import time
import cv2
import numpy as np
# import os
# import requests
# import imut


# photo = np.zeros((500,500,3),dtype='uint8')
#
# # photo[50:200,100:150] = 229,235,52
# cv2.rectangle(photo,(50,50),(150,150),(229,235,52),thickness=5)
# cv2.line(photo,(100,100),(100,0),(229,235,52), thickness=3)
# cv2.circle(photo,(photo.shape[1] // 2,photo.shape[0] // 2),70,(229,235,52),thickness=cv2.FILLED)
# cv2.putText(photo,'Hello World!',(100,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
#
#
#
# cv2.imshow('Resulth',photo)
# cv2.waitKey(0)



# Web camera for Phone

# url = 'http://192.168.1.2:8080/shot.jpg'
# ai = cv2.CascadeClassifier('Face_ai.xml')
# pTime = 0
#
# while True:
#     img_resp = requests.get(url)
#     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     img = cv2.imdecode(img_arr, -1)
#     img = imutils.resize(img, width=1000, height=1800)
#     cv2.imshow("Android_cam", img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     result = ai.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
#
#     for (x,y,w,h) in result:
#         cv2.rectangle(img,(x,y),(x + w,y + h),(255,11,2),thickness=9)
#
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img,f'FPS -->  {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),3)
#
#     if cv2.waitKey(1) == 27:
#         break
#
# cv2.destroyAllWindows()


# img = cv2.imread('images/NEURAL_NETWORKS-2.jpg')
# new_img = np.zeros(img.shape,dtype='uint8')
#
#
#
#
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img,(5,5),0)
# img = cv2.Canny(img,100,150)
# con,hir = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#
# cv2.drawContours(new_img,con,-1,(145,65,190),1)
#
# cv2.imshow('Resulth',new_img)
#
# cv2.waitKey(0)


# img = cv2.flip(img,1)

# def func(img_param,angle):
#     height,width = img.shape[:2]
#     point = (height // 2,width // 2)
#
#     mat = cv2.getRotationMatrix2D(point,angle,1)
#     return cv2.warpAffine(img_param,mat,(width,height))

# img = func(img,45)


# def transform(img_param,x,y):
#     mat = np.float32([[1,0,x],[0,1,y]])
#     return cv2.warpAffine(img_param,mat,(img_param.shape[1],img_param.shape[0]))
#
# img = transform(img,50,50)

# img = cv2.imread('images/NEURAL_NETWORKS-2.jpg')
# # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#
# r,g,b = cv2.split(img)
#
# cv2.imshow('Res',b)
# cv2.waitKey(0)

# img = np.zeros((500,500),dtype='uint8')
#
# circle = cv2.circle(img.copy(),(100,100),50,255,-1)
# square = cv2.rectangle(img.copy(),(300,300),(400,400),255,-1)
#
# cv2.imshow('Result',square)
# cv2.waitKey(0)


# img = cv2.imread('images/face3.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# faces = cv2.CascadeClassifier('Face_ai.xml')
#
# result = faces.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=6)
#
# for (x,y,w,h) in result:
#     cv2.rectangle(img,(x,y),(x + w,y + h),(0,255,0),thickness=2)
#
# cv2.imshow('Res',img)
# cv2.waitKey(0)



ai = cv2.CascadeClassifier('Face_ai.xml')

vid = cv2.VideoCapture('videos/VID-20231127-WA0001.mp4')
pTime = 0
frame_count = 0
des_width = 1000
des_height = 1000
saved_photo = np.ones((550,550,3),dtype='uint8')
# saved_photo[:] = 255,0,0

# cv2.imshow('resul',saved_photo)
# cv2.waitKey(0)


while 1:
    frame, img = vid.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = ai.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
    if frame:
        frame_count += 1
        resized_frame = cv2.resize(frame, (500, 500))
        image_filename = f'frame_{frame_count}.png'
        cv2.imwrite(image_filename, frame)
        print(f'Saved: {image_filename}')

        cv2.imshow('Face_photo', frame)

    for (x,y,w,h) in result:
        cv2.rectangle(img,(x,y),(x + w,y + h),(255,11,2),thickness=9)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img,f'FPS -->  {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),3)

    cv2.imshow('Testing',img)
    if cv2.waitKey(1) & 0x23 == ord('h'):
        break

cv2.imshow('Test',img)
cv2.destroyAllWindows()










