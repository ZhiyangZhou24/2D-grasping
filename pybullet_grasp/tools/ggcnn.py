# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：test-sgdn.py
@ IDE ：PyCharm
@ Function : sgdn测试类
"""

import cv2
import os
from numpy.lib.function_base import append
import torch
import time
import math
from skimage.feature import peak_local_max
import numpy as np
from skimage.draw import line


def ptsOnRect(pts):
    """
    获取矩形框上五条线上的点
    五条线分别是：四条边缘线，1条对角线
    pts: np.array, shape=(4, 2) (row, col)
    """
    rows1, cols1 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[1, 0]), int(pts[1, 1]))
    rows2, cols2 = line(int(pts[1, 0]), int(pts[1, 1]), int(pts[2, 0]), int(pts[2, 1]))
    rows3, cols3 = line(int(pts[2, 0]), int(pts[2, 1]), int(pts[3, 0]), int(pts[3, 1]))
    rows4, cols4 = line(int(pts[3, 0]), int(pts[3, 1]), int(pts[0, 0]), int(pts[0, 1]))
    rows5, cols5 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[2, 0]), int(pts[2, 1]))

    rows = np.concatenate((rows1, rows2, rows3, rows4, rows5), axis=0)
    cols = np.concatenate((cols1, cols2, cols3, cols4, cols5), axis=0)
    return rows, cols

def ptsOnRotateRect(pt1, pt2, w):
    """
    绘制矩形
    已知图像中的两个点（x1, y1）和（x2, y2），以这两个点为端点画线段，线段的宽是w。这样就在图像中画了一个矩形。
    pt1: [row, col] 
    w: 单位像素
    img: 绘制矩形的图像, 单通道
    """
    y1, x1 = pt1
    y2, x2 = pt2

    if x2 == x1:
        if y1 > y2:
            angle = math.pi / 2
        else:
            angle = 3 * math.pi / 2
    else:
        tan = (y1 - y2) / (x2 - x1)
        angle = np.arctan(tan)

    points = []
    points.append([y1 - w / 2 * np.cos(angle), x1 - w / 2 * np.sin(angle)])
    points.append([y2 - w / 2 * np.cos(angle), x2 - w / 2 * np.sin(angle)])
    points.append([y2 + w / 2 * np.cos(angle), x2 + w / 2 * np.sin(angle)])
    points.append([y1 + w / 2 * np.cos(angle), x1 + w / 2 * np.sin(angle)])
    points = np.array(points)

    # 方案1，比较精确，但耗时
    # rows, cols = polygon(points[:, 0], points[:, 1], (10000, 10000))	# 得到矩形中所有点的行和列

    # 方案2，速度快
    return ptsOnRect(points)	# 得到矩形中所有点的行和列

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    img:    rgb图像
    grasps: list()	元素是 [row, col, angle, width]
    mode:   line or region
    """
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            width = width / 2

            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)


        else:
            img[row, col] = [color_b, color_g, color_r]
        #

    return img

def drawRect(img, rect):
    """
    绘制矩形
    rect: [x1, y1, x2, y2]
    """
    print(rect)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


def depth2Gray(im_depth):
    """
    将深度图转至8位灰度图
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


def input_img(img, out_size=300):
    """
    对图像进行裁剪，保留中间(320, 320)的图像
    :param file: rgb文件
    :return: 直接输入网络的tensor, 裁剪区域的左上角坐标
    """

    assert img.shape[0] >= out_size and img.shape[1] >= out_size, '输入的深度图必须大于等于(320, 320)'

    # 裁剪中间图像块
    crop_x1 = int((img.shape[1] - out_size) / 2)
    crop_y1 = int((img.shape[0] - out_size) / 2)
    crop_x2 = crop_x1 + out_size
    crop_y2 = crop_y1 + out_size
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # 归一化
    img = np.clip(img - img.mean(), -1., 1.).astype(np.float32)

    # 调整顺序，和网络输入一致
    tensor = torch.from_numpy(img[np.newaxis, np.newaxis, :, :])  # np转tensor

    return tensor, crop_x1, crop_y1


def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    :param array: 二维array
    :param thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs


def collision_detection(pt, dep, angle, depth_map, finger_l1, finger_l2):
    """
    碰撞检测
    pt: (row, col)
    angle: 抓取角 弧度
    depth_map: 深度图
    finger_l1 l2: 像素长度

    return:
        True: 无碰撞
        False: 有碰撞
    """
    row, col = pt

    # 两个点
    row1 = int(row - finger_l2 * math.sin(angle))
    col1 = int(col + finger_l2 * math.cos(angle))
    
    # 在截面图上绘制抓取器矩形
    # 检测截面图的矩形区域内是否有1
    rows, cols = ptsOnRotateRect([row, col], [row1, col1], finger_l1)

    if np.min(depth_map[rows, cols]) > dep:   # 无碰撞
        return True
    return False    # 有碰撞

def getGraspDepth(camera_depth, grasp_row, grasp_col, grasp_angle, grasp_width, finger_l1, finger_l2):
    """
    根据深度图像及抓取角、抓取宽度，计算最大的无碰撞抓取深度（相对于物体表面的下降深度）
    此时抓取点为深度图像的中心点
    camera_depth: 位于抓取点正上方的相机深度图
    grasp_angle：抓取角 弧度
    grasp_width：抓取宽度 像素
    finger_l1 l2: 抓取器尺寸 像素长度

    return: 抓取深度，相对于相机的深度
    """
    # grasp_row = int(camera_depth.shape[0] / 2)
    # grasp_col = int(camera_depth.shape[1] / 2)
    # 首先计算抓取器两夹爪的端点
    k = math.tan(grasp_angle)

    grasp_width /= 2
    if k == 0:
        dx = grasp_width
        dy = 0
    else:
        dx = k / abs(k) * grasp_width / pow(k ** 2 + 1, 0.5)
        dy = k * dx
    
    pt1 = (int(grasp_row - dy), int(grasp_col + dx))
    pt2 = (int(grasp_row + dy), int(grasp_col - dx))

    # 下面改成，从抓取线上的最高点开始向下计算抓取深度，直到碰撞或达到最大深度
    rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])   # 获取抓取线路上的点坐标
    min_depth = np.min(camera_depth[rr, cc])
    # print('camera_depth[grasp_row, grasp_col] = ', camera_depth[grasp_row, grasp_col])

    grasp_depth = min_depth + 0.003
    while grasp_depth < min_depth + 0.05:
        if not collision_detection(pt1, grasp_depth, grasp_angle, camera_depth, finger_l1, finger_l2):
            return grasp_depth - 0.003
        if not collision_detection(pt2, grasp_depth, grasp_angle + math.pi, camera_depth, finger_l1, finger_l2):
            return grasp_depth - 0.003
        grasp_depth += 0.003

    return grasp_depth

