import cv2
import numpy as np

x1 = np.random.uniform(0.2, 0.8)
y1 = np.random.uniform(-0.3, 0.3)

image1 = cv2.imread('/home/ljm/data/objects_depth.png')


image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image1, contours, -1, (0, 255, 0), 2)

for contour in contours:
    print(cv2.minAreaRect(contour))

cv2.imshow('img', image)
cv2.waitKey(0)
