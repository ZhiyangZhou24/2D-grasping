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
sys.path.append('/home/zzy/workspace/grasp/gitrepo/2D-grasping')
from pybullet_grasp.utils.simEnv import SimEnv
import pybullet_grasp.utils.tool as tool
import pybullet_grasp.utils.panda_sim_grasp_gripper as gripper_sim
from pybullet_grasp.utils.camera import Camera
from ggcnn import  drawGrasps, getGraspDepth
from grasp_generator_bullet import inferencer
import matplotlib.pyplot as plt 

FINGER_L1 = 0.015
FINGER_L2 = 0.005



def run(database_path, start_idx, objs_num,network):
    cid = p.connect(p.GUI)  # 连接服务器
    gripper = gripper_sim.GripperSimAuto(p, [1, 0, 0.5])  # 初始化抓取器
    env = SimEnv(p, database_path, gripper.gripperId) # 初始化虚拟环境类
    camera = Camera()   # 初始化相机类
   
    success_grasp = 0
    sum_grasp = 0
    tt = 5
    # 按照预先保存的位姿加载多物体
    env.loadObjsInURDF(start_idx, objs_num)
    t = 0
    continue_fail = 0
    while True:
        # 等物体稳定
        for _ in range(240*10):
            p.stepSimulation()
        # 渲染深度图
        gripper.resetGripperPose([1, 0, 0.1], 0, 0)
        env.movecamera(0, 0)
        camera_depth = env.renderCameraDepthImage()
        camera_rgb = env.renderCameraRGBImage()
        camera_depth = env.add_noise(camera_depth)
        
        # plt.plot(camera_rgb)

        # 预测抓取
        row, col, grasp_angle, grasp_width_pixels ,q_img = network.predict(color_img = camera_rgb,depth_img = camera_depth)
        # row, col, grasp_angle, grasp_width_pixels,_ = network.predict(camera_depth, input_size=300)
        grasp_width_pixels *= 1.5
        grasp_width = camera.pixels_TO_length(grasp_width_pixels, camera_depth[row, col])

        grasp_x, grasp_y, grasp_z = camera.img2world([col, row], camera_depth[row, col]) # [x, y, z]
        finger_l1_pixels = camera.length_TO_pixels(FINGER_L1, camera_depth[row, col])
        finger_l2_pixels = camera.length_TO_pixels(FINGER_L2, camera_depth[row, col])
        grasp_depth = getGraspDepth(camera_depth, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
        grasp_z = max(0.7 - grasp_depth, 0)
        
        # print('*' * 100)
        # print('grasp pose:')
        # print('grasp_x = ', grasp_x)
        # print('grasp_y = ', grasp_y)
        # print('grasp_z = ', grasp_z)
        # print('grasp_depth = ', grasp_depth)
        # print('grasp_angle = ', grasp_angle)
        # print('grasp_width = ', grasp_width)
        # print('*' * 100)

        # 绘制抓取配置
        im_rgb = tool.depth2Gray3(camera_depth)
        im_grasp = drawGrasps(im_rgb, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')  # 绘制预测结果
        cv2.imshow('im_grasp', im_grasp)
        cv2.moveWindow('im_grasp',2000,1000)
        cv2.waitKey(30)

        dist = 0.4
        offset = 0.111    # 机械手的偏移距离
        gripper.resetGripperPose([grasp_x, grasp_y, grasp_z+dist+offset], grasp_angle, grasp_width/2)

        # 抓取
        t = 0
        while True:
            p.stepSimulation()
            t += 1
            if t % tt == 0:
                time.sleep(1./240.)
            
            if gripper.step(dist):  # 机器人抓取
                t = 0
                break

        # 遍历所有物体，只要有物体位于指定的坐标范围外，就认为抓取正确
        # sum_grasp += 1
        # if env.evalGraspAndRemove(z_thresh=0.3):
        #     success_grasp += 1
        #     continue_fail = 0
        #     if env.num_urdf == 0:
        #         p.disconnect()
        #         return success_grasp, sum_grasp
        # else:
        #     continue_fail += 1
        #     if continue_fail == 5:
        #         p.disconnect()
        #         return success_grasp, sum_grasp
        if env.evalGraspAndRemove(z_thresh=0.3):
            success_grasp = 1
            p.disconnect()
        else:
            success_grasp = 0
            p.disconnect()
        return success_grasp
                
        gripper.setGripper(0.04)    # 张开机械手



if __name__ == "__main__":

    # ggcnn = GGCNNNet('/home/zzy/workspace/grasp/wdx_code/ggcnn/output/models/221202_2243_wangdexin_test/epoch_0359_acc_0.7209.pth', device="cpu")    # 初始化ggcnn
    resu_model = '/home/zzy/workspace/grasp/2D-grasping-seres/logs/test_resu_cornell/230202_0631_resu_32b_rgbd_1000_ranger_img/epoch_24_iou_0.9816'
    grc_model ='/home/zzy/workspace/grasp/gitrepo/2D-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    grcnn_inferencer = inferencer(grc_model,use_depth=1,use_rgb=1,input_size=224,save_img = False)

    start_idx = 3       # 加载物体的起始索WW引


    objs_num = 1     # 场景的物体数量

    grasp_num = 20


    database_path = '/home/zzy/workspace/grasp/gitrepo/2D-grasping/pybullet_grasp/myModel/objs'
    success = 0
    success_num = 0
    for i in range(grasp_num):
        success = run(database_path, start_idx - 1, objs_num,grcnn_inferencer)
        if success == 1:
            success_num+=1
    print('\n>>>>>>>>>>>>>>>>>>>> Success Rate: {}/{}={}'.format(success_num, grasp_num, success_num/grasp_num))              
    
