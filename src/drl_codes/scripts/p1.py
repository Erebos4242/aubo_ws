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


def judge_start_point():
    _, image = env.get_img()
    cv2.imwrite('/home/ljm/data/objects_depth.png', image)
    _, binary = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('img', binary)
    cv2.waitKey(0)


judge_start_point()
