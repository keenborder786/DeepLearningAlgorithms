# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:10:05 2019

@author: MMOHTASHIM
"""

import cv2
import numpy as np

img=cv2.imread("bookpage.jpg")

retval,threshold=cv2.threshold(img,12,255,cv2.THRESH_BINARY)
grayscaled=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval,threshold2=cv2.threshold(grayscaled,12,255,cv2.THRESH_BINARY)
gaus=cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
retval2,threshold2 = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("orginal",img)
cv2.imshow("threshold",threshold)
cv2.imshow("threshold",threshold2)
cv2.imshow("gaus",gaus)
