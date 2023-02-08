"""
虚拟环境文件
初始化虚拟环境，加载物体，渲染图像，保存图像

(待写) ！！ 保存虚拟环境状态，以便离线抓取测试
"""

import pybullet as p
import pybullet_data
import time
import math
import os
import glob
import random
import cv2
import shutil
import numpy as np
import scipy.io as scio
import sys
import scipy.stats as ss
import skimage.transform as skt
from pybullet_object_models import ycb_objects
sys.path.append('/home/zzy/workspace/grasp/gitrepo/2D-grasping/pybullet_grasp')
# from utils.mesh import Mesh
# import utils.tool as tool
# from utils.camera import Camera
from PIL import Image
# 图像尺寸
IMAGEWIDTH = 640
IMAGEHEIGHT = 480

nearPlane = 0.01
farPlane = 10

fov = 60    # 垂直视场 图像高tan(30) * 0.7 *2 = 0.8082903m
aspect = IMAGEWIDTH / IMAGEHEIGHT



def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

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


class SimEnv(object):
    """
    虚拟环境类
    """
    def __init__(self, bullet_client, path, gripperId=None):
        """
        path: 模型路径
        """
        self.p = bullet_client
        # self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, solverResidualThreshold=0, enableFileCaching=0)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加路径
        self.planeId = self.p.loadURDF("plane.urdf", [0, 0, 0])  # 加载地面    
        self.trayId = self.p.loadURDF('myModel/tray/urdfs/setup/plane.urdf', [0, 0, 0.01])
        self.p.setGravity(0, 0, -5) # 设置重力
        self.flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.gripperId = gripperId

        # 加载相机
        self.movecamera(0, 0)
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        # 读取path路径下的list.txt文件
        list_file = os.path.join(path, 'ycbs.txt')
        if not os.path.exists(list_file):
            raise shutil.Error
        self.urdfs_list = []
        with open(list_file, 'r') as f:
            while 1:
                line = f.readline()
                # prinst(line[:-1])
                if not line:
                    break
                self.urdfs_list.append(os.path.join(ycb_objects.getDataPath(), line[:-1] ,"model.urdf") )
        
        self.num_urdf = 0
        self.urdfs_id = []  # 存储由pybullet系统生成的模型id
        self.objs_id = []   # 存储模型在文件列表中的索引，注意，只在path为str时有用
        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]

    
    def _urdf_nums(self):
        return len(self.urdfs_list)
    

    def movecamera(self, x, y, z=0.7):
        """
        移动相机至指定位置
        x, y: 世界坐标系中的xy坐标
        """
        self.viewMatrix = self.p.computeViewMatrix([x, y, z], [x, y, 0], [0, 1, 0])   # 相机高度设置为0.7m



    # 加载单物体
    def loadObjInURDF(self, urdf_file, idx, render_n=0):
        """
        以URDF的格式加载单个obj物体

        urdf_file: urdf文件
        idx: 物体id， 等于-1时，采用file
        render_n: 当前物体的渲染次数，根据此获取物体的朝向
        """
        # 获取物体文件
        if idx >= 0:
            self.urdfs_filename = [self.urdfs_list[idx]]
            self.objs_id = [idx]
        else:
            self.urdfs_filename = [urdf_file]
            self.objs_id = [-1]
        self.num_urdf = 1


        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []

        # 方向
        # baseEuler = [self.EulerRPList[render_n][0], self.EulerRPList[render_n][1], 0]
        # baseEuler = [self.EulerRPList[render_n][0], self.EulerRPList[render_n][1], random.uniform(0, 2*math.pi)]
        baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
        baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
        # baseOrientation = [0, 0, 0, 1]    # 固定方向

        # 随机位置
        pos = 0.05
        # basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.1, 0.4)] 
        # basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), -1*min_z] 
        basePosition = [0, 0, 0] # 固定位置

        # 加载物体
        urdf_id = self.p.loadURDF(self.urdfs_filename[0], basePosition, baseOrientation)    

        # 获取xyz和scale信息
        inf = self.p.getVisualShapeData(urdf_id)[0]

        self.urdfs_id.append(urdf_id)
        self.urdfs_xyz.append(inf[5]) 
        self.urdfs_scale.append(inf[3][0]) 
    
    
    # 加载多物体
    def loadObjsInURDF(self, idx, num):
        """
        以URDF的格式加载多个obj物体

        num: 加载物体的个数
        idx: 开始的id
            idx为负数时，随机加载num个物体
            idx为非负数时，从id开始加载num个物体
        """
        assert idx >= 0 and idx < len(self.urdfs_list)
        self.num_urdf = num

        # 获取物体文件
        if (idx + self.num_urdf - 1) > (len(self.urdfs_list) - 1):     # 这段代码主要针对加载多物体的情况
            self.urdfs_filename = self.urdfs_list[idx:]
            self.urdfs_filename += self.urdfs_list[:2*self.num_urdf-len(self.urdfs_list)+idx]
            self.objs_id = list(range(idx, len(self.urdfs_list)))
            self.objs_id += list(range(self.num_urdf-len(self.urdfs_list)+idx))
        else:
            self.urdfs_filename = self.urdfs_list[idx:idx+self.num_urdf]
            self.objs_id = list(range(idx, idx+self.num_urdf))
        
        # print('self.urdfs_filename = \n', self.urdfs_filename)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        for i in range(self.num_urdf):
            # 随机位置
            pos = 0.1
            # basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.2, 0.3)] 
            basePosition = [0.0, 0.0, 0.2] # 固定位置

            # 随机方向
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            # baseOrientation = [0, 0, 0, 1]    # 固定方向
            
            # 加载物体
            urdf_id = self.p.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation)    
            # 使物体和机械手可以碰撞
            # 机械手默认和所有物体不进行碰撞
            if self.gripperId is not None:
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 0, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 1, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 2, 1)

            # 获取xyz和scale信息
            inf = self.p.getVisualShapeData(urdf_id)[0]

            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5]) 
            self.urdfs_scale.append(inf[3][0]) 
            
            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break


    def evalGrasp(self, z_thresh):
        """
        验证抓取是否成功
        如果某个物体的z坐标大于z_thresh，则认为抓取成功
        """
        for i in range(self.num_urdf):
            offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[i])
            if offset[2] >= z_thresh:
                return True
        print('!!!!!!!!!!!!!!!!!!!!! 失败 !!!!!!!!!!!!!!!!!!!!!')
        return False

    def evalGraspAndRemove(self, z_thresh):
        """
        验证抓取是否成功，并删除抓取的物体
        如果某个物体的z坐标大于z_thresh，则认为抓取成功
        """
        for i in range(self.num_urdf):
            offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[i])
            if offset[2] >= z_thresh:
                self.removeObjInURDF(i)
                return True
        print('!!!!!!!!!!!!!!!!!!!!! 失败 !!!!!!!!!!!!!!!!!!!!!')
        return False
    

    def resetObjsPoseRandom(self):
        """
        随机重置物体的位置
        path: 存放物体位姿文件的文件夹
        """
        # 读取path下的objsPose.mat文件
        for i in range(self.num_urdf):
            pos = 0.1
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.3, 0.6)]
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            self.p.resetBasePositionAndOrientation(self.urdfs_id[i], basePosition, baseOrientation)

            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break


    def removeObjsInURDF(self):
        """
        移除所有objs
        """
        for i in range(self.num_urdf):
            self.p.removeBody(self.urdfs_id[i])
        self.num_urdf = 0
        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        self.urdfs_filename = []
        self.objs_id = []

    def removeObjInURDF(self, i):
        """
        移除指定的obj
        """
        self.num_urdf -= 1
        self.p.removeBody(self.urdfs_id[i])
        self.urdfs_id.pop(i)
        self.urdfs_xyz.pop(i)
        self.urdfs_scale.pop(i)
        self.urdfs_filename.pop(i)
        self.objs_id.pop(i)


    def renderCameraDepthImage(self):
        """
        渲染计算抓取配置所需的图像
        """
        # 渲染图像
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        dep = img_camera[3]    # depth data

        # 获取深度图像
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # 单位 m
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m
        return im_depthCamera

    def renderCameraRGBImage(self):
        """
        渲染计算抓取配置所需的图像
        """
        # 渲染图像
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        rgba_raw = img_camera[2]    # color data RGB
        image_without_alpha = rgba_raw[:,:,:3]

        return image_without_alpha

    def renderCameraMask(self):
        """
        渲染计算抓取配置所需的图像
        """
        # 渲染图像
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        # rgba = img_camera[2]    # color data RGB
        # dep = img_camera[3]    # depth data
        mask = img_camera[4]    # mask data

        # 获取分割图像
        im_mask = np.reshape(mask, (h, w)).astype(np.uint8)
        im_mask[im_mask > 2] = 255
        return im_mask


    def gaussian_noise(self, im_depth):
        """
        在image上添加高斯噪声，参考dex-net代码

        im_depth: 浮点型深度图，单位为米
        """
        gamma_shape = 1000.00
        gamma_scale = 1 / gamma_shape
        gaussian_process_sigma = 0.002  # 0.002
        gaussian_process_scaling_factor = 8.0   # 8.0

        im_height, im_width = im_depth.shape
        
        # 1
        # mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=1) # 生成一个接近1的随机数，shape=(1,)
        # mult_samples = mult_samples[:, np.newaxis]
        # im_depth = im_depth * np.tile(mult_samples, [im_height, im_width])  # 把mult_samples复制扩展为和camera_depth同尺寸，然后相乘
        
        # 2
        gp_rescale_factor = gaussian_process_scaling_factor     # 4.0
        gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
        gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0
        gp_num_pix = gp_sample_height * gp_sample_width     # im_height * im_width / 16.0
        gp_sigma = gaussian_process_sigma
        gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)  # 生成(均值为0，方差为scale)的gp_num_pix个数，并reshape
        # print('高斯噪声最大误差:', gp_noise.max())
        gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")  # resize成图像尺寸，bicubic为双三次插值算法
        # gp_noise[gp_noise < 0] = 0
        # camera_depth[camera_depth > 0] += gp_noise[camera_depth > 0]
        im_depth += gp_noise

        return im_depth

    def add_noise(self, img):
        """
        添加高斯噪声和缺失值
        """
        img = self.gaussian_noise(img)    # 添加高斯噪声
        # 补全
        # img = inpaint(img, missing_value=0)
        return img