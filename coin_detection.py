import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("images/Coins2.png")
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=10, minRadius=50, maxRadius=115)

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(hsv[:,:,::-1])
plt.subplot(1,3,2)
plt.title("Coins Detected")
plt.imshow(hsv_cv[:,:,::-1])


