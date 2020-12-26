import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read coin images
image = cv2.imread('images/Coins4.jpg', cv2.IMREAD_COLOR)
image1 = cv2.imread('images/Coins1.png', cv2.IMREAD_COLOR)
image2 = cv2.imread('images/Coins2.png', cv2.IMREAD_COLOR)
image3 = cv2.imread('images/Coins3.png', cv2.IMREAD_COLOR)
imageA = cv2.imread('images/CoinsA.png', cv2.IMREAD_COLOR)
imageB = cv2.imread('images/CoinsB.png', cv2.IMREAD_COLOR)


def hough_find_circle(image, dp, minDist, param1, param2, minRadius, maxRadius):
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image. By reducing noise it improves the edge detection
    image_blur = cv2.medianBlur(image_gray, 5)
    # Apply hough transform on the image
    circles = cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius)
    
    return circles

def draw_circles(image, circles):
    # Create a copy of the orginal image to avoid changing it
    new_image = image.copy()
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(new_image, (i[0], i[1]), i[2], (0, 255, 0), 7)
            # Draw inner circle
            cv2.circle(new_image, (i[0], i[1]), 2, (0, 0, 255), 5)
    else:
        print("There is no circle coordinates!")
        return None
    return circles

# CoinsA Hough config:
# cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=10, minRadius=30, maxRadius=50)
    
coinA_circles = hough_find_circle(imageA, dp=1, minDist=100, param1=250, param2=10, minRadius=50, maxRadius=115)
coinA = draw_circles(imageA, coinA_circles)

# CoinsB Hough config:
# cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, 1, 525, param1=250, param2=10, minRadius=100, maxRadius=300)

coinB_circles = hough_find_circle(imageB, dp=1, minDist=525, param1=250, param2=10, minRadius=100, maxRadius=300)
coinB = draw_circles(imageB, coinB_circles)

# Coins1 Hough config:
# cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, 1, 525, param1=250, param2=10, minRadius=100, maxRadius=300)

#coin1_circles = hough_find_circle(image1, dp=1, minDist=525, param1=250, param2=10, minRadius=100, maxRadius=300)
#coin1 = draw_circles(image1, coin1_circles)

plt.figure(figsize=[40,24])
#plt.subplot(231);plt.imshow(image[:,:,::-1], cmap='gray', vmin=0, vmax=1);
#plt.title("Original Image");
plt.subplot(232);plt.imshow(coinA[:,:,::-1], cmap='gray', vmin=0, vmax=1);
plt.title("CoinsA Detected");
plt.subplot(233);plt.imshow(coinB[:,:,::-1], cmap='gray', vmin=0, vmax=1);
plt.title("CoinsB Detected");


