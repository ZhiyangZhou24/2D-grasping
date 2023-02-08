'''
Description: 生成模型库中的list.txt文件
Author: wangdx
Date: 2021-11-10 15:10:24
LastEditTime: 2021-11-20 22:49:31
'''

import os
import glob
import random

path = '/home/zzy/workspace/grasp/gitrepo/pybullet_grasp/myModel/test'

files = glob.glob(os.path.join(path, '*', '*.urdf'))
random.shuffle(files)

txt = open(path + '/list.txt', 'w+')
for f in files:
    fname = os.path.basename(f)
    pre_fname = os.path.basename(os.path.dirname(f))
    txt.write(pre_fname + '/' + fname[:-5] + '\n')
txt.close()
print('done')

    
