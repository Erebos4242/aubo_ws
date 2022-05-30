#!/usr/bin/env python3
import gym
import math
import random as randomR
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image as PILimage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as torch_tran
from myrobotenv import *
import time

env = RobotSim()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = torch_tran.Compose([torch_tran.ToPILImage(),
                    torch_tran.ToTensor()])

def get_screen():
    _, image = env.get_img()
    image = image[:, 140:500]
    image = image[60: 420]

    cv2.imwrite('/home/ljm/data/temp.png', image)
    
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32) / 255
    image = torch.from_numpy(image)
    # Resize, and add a batch dimension (BCHW)
    return resize(image).unsqueeze(0)


screen_height, screen_width = 360, 360
n_actions = 30
dqn_net = DQN(screen_height, screen_width, n_actions).to(device)
dqn_net.load_state_dict(torch.load('/home/ljm/data/saved_net/dqt_push/model.txt'))
dqn_net.eval()

test_num = 100
for i in range(test_num):

    print(f'test num: {i}')

    # Initialize the environment and state
    env.reset()
    state = get_screen()
    
    done_step = 0
    for t in count():
        # Select and perform an action
        action = dqn_net(state).max(1)[1].view(1, 1)
        done, reward = env.step(action.item())
        print(f'action: {action.item()}, reward: {reward}')

        reward = torch.tensor([reward], device=device)
        
        # Observe new state
        if not done:
            next_state = get_screen()
        else:
            next_state = None

        done_step += 1
        if done or done_step >= 20:
            print(f'total step: {done_step}')
            break

print('Complete')
