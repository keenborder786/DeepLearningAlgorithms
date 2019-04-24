# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:56:53 2019

@author: MMOHTASHIM
"""

import cv2
import numpy as np
img1=cv2.imread("3D-Matplotlib.png")
img2=cv2.imread("mainlogo.png")
cv2.imshow("img",img2)
#add=img1+img2#opqaueness remains

##it added which made the image too bright which made the screen white

#add=cv2.add(img1,img2)

#wieghted=cv2.addWeighted(img1,0.6,img2,0.4,0)

rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]
cv2.imshow("roi",roi)
img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask=cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV) #if img pixel is above 220 make it 255 otherwise make it black(inverse of this since
cv2.imshow("m",mask)
#we are using inv_binary)
#cv2.imshow('mask',mask)

mask_inv=cv2.bitwise_not(mask)
cv2.imshow("mask_inv",mask_inv)

img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow("img1_bg",img1_bg)
img2_fg = cv2.bitwise_and(img2,img2,mask=mask)
cv2.imshow("img2",img2_fg)
dst=cv2.add(img1_bg,img2_fg)
img1[0:rows,0:cols]=dst
#cv2.imshow("res",img1)


                       



#cv2.imshow("add",wieghted)