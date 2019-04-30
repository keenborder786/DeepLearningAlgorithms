# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:31:34 2019

@author: MMOHTASHIM
"""

import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    #hsv hue set value
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_red=np.array([50,0,0])
    upper_red=np.array([255,255,255])
    
    
    kern=np.ones((15,15),np.float32)/255
    
    mask=cv2.inRange(hsv,lower_red,upper_red)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    
    #below are jut different types of filter applied on our res
    smooth=cv2.filter2D(res,-1,kern)
    median=cv2.medianBlur(res,15)
    bilateral=cv2.bilateralFilter(res,15,75,75)
    blur=cv2.GaussianBlur(res,(15,15),0)
    
    
    cv2.imshow('frame',frame)
   # cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('blur',blur)
    cv2.imshow('smooth',smooth)
    
    
    k=cv2.waitKey(5)& 0xff
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()
    
    