import cv2


states = cv2.imread('/home/ljm/data/push_ws.png')


class TestEnv():
    def __init__(self):
        pass
        
    def get_img(self):
        return 0, states

    def step(self, a):
        return False, 1

    def reset(self):
        return