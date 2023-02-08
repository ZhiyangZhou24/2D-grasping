'''
Description: 为现有的obj创建collision obj 模型
Author: wangdx
Date: 2021-11-20 21:53:02
LastEditTime: 2021-11-21 12:58:21
'''

import os
import pybullet as p


path = '/home/zzy/workspace/grasp/gitrepo/pybullet_grasp/myModel/test/meshes'

p.connect(p.DIRECT)
files = os.listdir(path)
for file in files:
    print('processing ...', file)
    name_in = os.path.join(path, file)
    name_out = os.path.join(path, file.replace('.obj', '_col.obj'))
    name_log = "log.txt"
    p.vhacd(name_in, name_out, name_log)
