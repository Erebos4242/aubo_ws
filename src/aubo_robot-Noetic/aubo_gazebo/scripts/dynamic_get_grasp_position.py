#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Header
from aubo_msgs.msg import GraspPosition
from image_frame_to_world_frame import image_frame_to_world_frame
from time import time


table_height = 1.5342424
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920


def get_position(img):

    new_img = bridge.imgmsg_to_cv2(img, "passthrough")
    depth_array = np.array(new_img, dtype=np.float32)
    depth = depth_array.copy()
    cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
    image = np.stack((depth_array*255,)*3, axis=-1)
    image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    object_position = []
    for index in range(len(contours)):
        rect = cv2.minAreaRect(contours[index])
        center_x, center_y = int(rect[0][1]), int(rect[0][0])
        h = depth[center_x][center_y]
        res = list(image_frame_to_world_frame(h, center_x, center_y))
        res[2] = 2.5 - table_height + (table_height - h) / 2

        if rect[1][0] >= rect[1][1]:
            res.append(rect[2])
        else:
            res.append(90 + rect[2])

        object_position += list(res)
        cv2.putText(image, "(%.3f, %.3f, %.3f, %.3f)" % (res[0], res[1], res[2], res[3]), (center_y - 220, center_x), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 1, cv2.LINE_AA)
      
    grasp_position_pub.publish(object_position)

    image_temp=Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='rgb8'
    image_temp.data=np.array(image).tostring()
    image_temp.header=header
    image_temp.step=IMAGE_WIDTH*3

    grasp_position_map_pub.publish(image_temp)
    


def main():
    rospy.init_node('dynamic_get_grasp_position')
    rospy.Subscriber("/camera/depth/image_raw", Image, get_position)
    rospy.spin()


if __name__ == "__main__":
    bridge = CvBridge()
    grasp_position_map_pub = rospy.Publisher('/grasp/map', Image, queue_size=10)
    grasp_position_pub = rospy.Publisher('/grasp/position', GraspPosition, queue_size=1)
    main()

