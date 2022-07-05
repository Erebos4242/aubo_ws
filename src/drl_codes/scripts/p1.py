#!/usr/bin/env python3
import torch
import torchvision.transforms as torch_tran
import numpy as np
import cv2
import time
import json
import os
from myrobotenv import *

env = RobotSim()


def judge_start_point(x, y):
    _, image_org = env.get_img()
    lenth_per_pixel = 0.00186
    x -= 0.5

    x_pixel = 240 - int(x / lenth_per_pixel)
    y_pixel = 320 + int(y / lenth_per_pixel)
    off_set = 6

    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV)

    cv2.circle(image, (y_pixel, x_pixel), 10, (255, 255, 255), 1)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    if image[x_pixel - off_set][y_pixel] == 255 or image[x_pixel + off_set][y_pixel] == 255 or \
            image[x_pixel][y_pixel - off_set] == 255 or image[x_pixel][y_pixel + off_set] == 255 or \
            image[x_pixel - 5][y_pixel - 5] == 255 or image[x_pixel + 5][y_pixel + 5] == 255 or \
            image[x_pixel - 5][y_pixel + 5] == 255 == 255 or image[x_pixel + 5][y_pixel - 5] == 255 == 255:
        return True
    else:
        return False


print(judge_start_point(0.63, -0.2))
