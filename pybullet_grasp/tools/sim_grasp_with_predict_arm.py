'''
Description: 
Author: wangdx
Date: 2021-09-06 22:09:19
LastEditTime: 2021-11-28 15:30:39
'''
"""
用于验证神经网络预测的抓取

运行流程：
(1) 加载物体和渲染的深度图
(2) 输入网络，获取预测抓取
(3) 相机移动至抓取点上方，计算最大的抓取深度
(4) 实施抓取
"""

import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import numpy as np
import sys
import scipy.io as scio
sys.path.append('/home/zzy/workspace/grasp/gitrepo/pybullet_grasp')
from utils.simEnv import SimEnv
import utils.tool as tool
import utils.panda_sim_grasp_arm as PandaSim
from utils.camera import Camera
from ggcnn.ggcnn import GGCNNNet, drawGrasps, drawRect, getGraspDepth

FINGER_L1 = 0.015
FINGER_L2 = 0.005


def run(database_path, start_idx, objs_num):
    cid = p.connect(p.GUI)  # 连接服务器
    panda = PandaSim.PandaSimAuto(p, [0, -0.6, 0])  # 初始化抓取器
    env = SimEnv(p, database_path, panda.pandaId) # 初始化虚拟环境类
    camera = Camera()   # 初始化相机类
    ggcnn = GGCNNNet('ggcnn/ckpt/epoch_0213_acc_0.6374.pth', device="cpu")    # 初始化ggcnn

    success_grasp = 0
    sum_grasp = 0
    tt = 1
    # 按照预先保存的位姿加载多物体
    env.loadObjsInURDF(start_idx, objs_num)
    t = 0
    continue_fail = 0
    while True:
        # 等物体稳定
        for _ in range(240*5):
            p.stepSimulation()
        # 渲染深度图
        camera_depth = env.renderCameraDepthImage()
        camera_depth = env.add_noise(camera_depth)

        # 预测抓取

        # 
        row, col, grasp_angle, grasp_width_pixels ,pos= ggcnn.predict(camera_depth, input_size=300)
        grasp_width_pixels = 200
        grasp_width = camera.pixels_TO_length(grasp_width_pixels, camera_depth[row, col])

        grasp_x, grasp_y, grasp_z = camera.img2world([col, row], camera_depth[row, col]) # [x, y, z]
        finger_l1_pixels = camera.length_TO_pixels(FINGER_L1, camera_depth[row, col])
        finger_l2_pixels = camera.length_TO_pixels(FINGER_L2, camera_depth[row, col])
        grasp_depth = getGraspDepth(camera_depth, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
        grasp_z = max(0.7 - grasp_depth, 0)
        
        print('*' * 100)
        print('grasp pose:')
        print('grasp_x = ', grasp_x)
        print('grasp_y = ', grasp_y)
        print('grasp_z = ', grasp_z)
        print('grasp_depth = ', grasp_depth)
        print('grasp_angle = ', grasp_angle)
        print('grasp_width = ', grasp_width)
        print('*' * 100)

        # 绘制抓取配置
        im_rgb = tool.depth2Gray3(camera_depth)
        im_grasp = drawGrasps(im_rgb, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')  # 绘制预测结果
        cv2.imshow('im_grasp', im_grasp)
        cv2.waitKey(30)

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
        sum_grasp += 1
        if env.evalGraspAndRemove(z_thresh=0.2):
            success_grasp += 1
            continue_fail = 0
            if env.num_urdf == 0:
                p.disconnect()
                return success_grasp, sum_grasp
        else:
            continue_fail += 1
            if continue_fail == 5:
                p.disconnect()
                return success_grasp, sum_grasp
        
        panda.setArmPos([0.5, -0.6, 0.2])



if __name__ == "__main__":
    start_idx = 0       # 加载物体的起始索引
    objs_num = 5       # 场景的物体数量
    database_path = '/home/zzy/workspace/grasp/gitrepo/pybullet_grasp/myModel/objs'
    success_grasp, all_grasp = run(database_path, start_idx, objs_num)
    print('\n>>>>>>>>>>>>>>>>>>>> Success Rate: {}/{}={}'.format(success_grasp, all_grasp, success_grasp/all_grasp))     
    print('\n>>>>>>>>>>>>>>>>>>>> Percent Cleared: {}/{}={}'.format(success_grasp, objs_num, success_grasp/objs_num))    
