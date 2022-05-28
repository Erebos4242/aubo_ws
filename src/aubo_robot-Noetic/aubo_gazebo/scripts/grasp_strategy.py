#!/usr/bin/env python
import rospy
from var import *


def main():
    rospy.init_node('grasp_strategy', anonymous=True)
    
    rospy.spin()
    

if __name__ == "__main__":
    main()