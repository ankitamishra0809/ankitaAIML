# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:06:56 2021

@author: cttc
"""

import cv2
import urllib
import numpy as np
import matplotlib.pyplot as plt


URL="http://10.65.156.140:8080/shot.jpg"
face_data="haarcascade_frontalface_default.xml"
classifier=cv2.CascadeClassifier(face_data)

data=[]
ret=True

while True:
    img_url=urllib.request.urlopen(URL)
    image=np.array(bytearray(img_url.read()),np.uint8)
    frame=cv2.imdecode(image,-1)
    
    faces=classifier.detectMultiScale(frame,1.5,5)
    if faces is not None:
        for x,y,w,h in faces:
            
            
            face_image=frame[y:y+h,x:x+w].copy()
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            if len(data)<=100:
                data.append(face_image)
            else:
                cv2.puttext(frame,'complete',(200,200),
                cv2.FONT_HERSHEY_COMPLEX,1,
                (255,0,0),2)
            
    cv2.imshow('capture',frame)
    if cv2.waitKey(1)==ord('q'):
        break
    

cv2.destroyAllWindows()

name=input("enter name: ")
c=0
for i in data:
    cv2.imwrite("images/"+name+'_'+str(c)+'.jpg',i)
    c=c+1
for i in range(0,12):
    plt.imshow(data[i])
    plt.show()
