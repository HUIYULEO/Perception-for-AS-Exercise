#! /user/bin/python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils 

bgr_img = cv2.imread("appletree.jpg")
b,g,r = cv2.split(bgr_img)       # get b,g,r
image = cv2.merge([r,g,b])
(h, w, d) = image.shape
img_apple = np.zeros(image.shape, np.uint8)

print("width={}, height={}, depth={}".format(w, h, d))
#plt.imshow(image)
#plt.show()

#(R, G, B) = image[400, 420]
#print("R={}, G={}, B={}".format(R, G, B))

for i in range(h):
    for j in range(w):
        (R,G,B) = image[i,j]
        img_apple[i,j] = (255,255,255)
        if (R > G and R > B):
            img_apple[i,j] = (R,G,B)
# plt.imshow(img_apple)
# plt.show()


# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap = 'gray')
# #edged = cv2.Canny(gray, 30, 150)
# #plt.imshow(edged, cmap='gray')
# plt.show()

apple = cv2.cvtColor(img_apple, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV)
lower_red = np.array([50, 50, 46])
upper_red = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red, upper_red)
result1 = cv2.bitwise_and(img_apple, img_apple, mask = mask1)
result2 = img_apple - result1
# plt.imshow(result1)
# plt.show()

gray = cv2.cvtColor(result1, cv2.COLOR_RGB2GRAY)
threshold = 10
threshold_value = 200
thresh = cv2.threshold(gray, threshold, threshold_value, cv2.THRESH_BINARY_INV)[1]
plt.imshow(thresh, cmap='gray')
# plt.imshow(gray, cmap = 'gray')
plt.show()

'''
titles=['r','g','b']
plt.figure(figsize = (16,4))
for i in range(3):
    channel = np.zeros_like(image)
    channel[:,:,i] = image[:,:,i]
    plt.subplot(1,3,i+1), plt.imshow(channel)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
'''
