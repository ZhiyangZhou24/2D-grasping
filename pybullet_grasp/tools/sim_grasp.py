'''
Description: 
Author: wangdx
Date: 2021-09-06 22:09:19
LastEditTime: 2021-11-28 15:30:39
'''

import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import numpy as np
import sys
import scipy.io as scio
sys.path.append('/home/zzy/workspace/grasp/gitrepo/2D-grasping/pybullet_grasp')
from utils.simEnv import SimEnv
import utils.tool as tool
import utils.panda_sim_grasp_arm as PandaSim
from utils.camera import Camera


def run(database_path, start_idx, objs_num):
    cid = p.connect(p.GUI)  # 连接服务器
    panda = PandaSim.PandaSimAuto(p, [0, -0.6, 0])  # 初始化抓取器
    env = SimEnv(p, database_path, panda.pandaId) # 初始化虚拟环境类

    tt = 1
    # 按照预先保存的位姿加载多物体
    env.loadObjsInURDF(start_idx, objs_num)
    t = 0
    continue_fail = 0
    while True:
        # 等物体稳定
        for _ in range(240*5):
            p.stepSimulation()

        grasp_x, grasp_y, grasp_z, grasp_angle, grasp_width = (0, 0, 0.01, 0, 0.08)

        # 抓取
        t = 0
        while True:
            p.stepSimulation()
            t += 1
            if t % tt == 0:
                time.sleep(1./240.)
            
            if panda.step([grasp_x, grasp_y, grasp_z], grasp_angle, grasp_width/2):
                t = 0
                break

        # 遍历所有物体，只要有物体位于指定的坐标范围外，就认为抓取正确
        if env.evalGraspAndRemove(z_thresh=0.2):
            if env.num_urdf == 0:
                p.disconnect()
                return
        
        panda.setArmPos([0.5, -0.6, 0.2])


if __name__ == "__main__":
    start_idx = 0       # 加载物体的起始索引
    objs_num = 5       # 场景的物体数量
    database_path = '/home/zzy/workspace/grasp/gitrepo/pybullet_grasp/myModel/objs'
    run(database_path, start_idx, objs_num)
