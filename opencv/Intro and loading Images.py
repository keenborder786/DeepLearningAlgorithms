# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:57:50 2019

@author: MMOHTASHIM
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


img=cv2.imread("watch.jpg",cv2.IMREAD_GRAYSCALE)
#IMREAD_COLOR=1
#IMREAD_GRAYSCALE=0
#IMREAD_UNCHANGED=-1

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img,cmap="gray",interpolation="bicubic")
plt.plot([50,100],[80,100],'c',linewidth=5)

cv2.imwrite("watchgray.png",img)