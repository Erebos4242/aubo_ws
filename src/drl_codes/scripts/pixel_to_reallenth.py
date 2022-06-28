import cv2
import numpy as np

image_dir = '/home/ljm/data/pixel_cal.png'
real_lenth = 0.5

image = cv2.imread(image_dir)
img_height, img_width, _ = image.shape
half_height, half_width = int(img_height / 2), int(img_width / 2)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY_INV)


for i in range(len(image)):
    if image[i][half_width] == 255:
        y_per_pixel = real_lenth / (img_height - i*2)
        break

for i in range(len(image)):
    if image[half_height][i] == 255:
        x_per_pixel = real_lenth / (img_width - i*2)
        break

print('x_per_pixel', x_per_pixel)
print('y_per_pixel', y_per_pixel)