#import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import math
import sys
#from PIL import Image

sys.path.append('/home/zzy/workspace/grasp/gitrepo/2D-grasping')
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData     
from utils.visualisation.plot import plot_results, save_results
from utils.dataset_processing.grasp import detect_grasps

from datetime import datetime

logging.basicConfig(level=logging.INFO)

class inferencer:
    def __init__(self,model,use_depth,use_rgb,input_size=224,save_img = False):
        # Load Network
        logging.info('Loading model from {}'.format(model))
        # Load Network
        self.args_force_cpu=False
        self.args_use_rgb = use_rgb
        self.args_use_depth = use_depth
        # path = '/home/zzy/workspace/grasp/gitrepo/pybullet_grasp/grcnn/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
        if self.args_force_cpu:
            self.net = torch.load(model,map_location='cpu')    # the argument  map_location='cpu'  must be added if u r using a CPU machine
        else:
            self.net = torch.load(model)
        logging.info('Done')

        # Get the compute device
        self.device = get_device(self.args_force_cpu)

        self.img_data = CameraData(width=640,
                          height=480,
                          output_size=input_size,
                          include_depth=self.args_use_depth, 
                          include_rgb=self.args_use_rgb)

        self.args_save = save_img

    def predict(self,color_img,depth_img):
        # Load image
        rgb=color_img
        depth=np.expand_dims(depth_img, axis=2)

        x, depth_img, rgb_img =self.img_data.get_data(rgb=rgb, depth=depth)
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.net.predict(xc)

            # post process, gaussian blur and synthesise angle image
            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

            # if self.args_save:
            #     save_results(
            #         rgb_img=self.img_data.get_rgb(rgb, False),
            #         depth_img=np.squeeze(self.img_data.get_depth(depth)),
            #         grasp_q_img=q_img,
            #         grasp_angle_img=ang_img,
            #         no_grasps=1,
            #         grasp_width_img=width_img
            #     )
            # else:
            #     fig = plt.figure(figsize=(10, 10))
            #     plot_results(fig=fig,
            #                         rgb_img=self.img_data.get_rgb(rgb, False),
            #                         grasp_q_img=q_img,
            #                         grasp_angle_img=ang_img,
            #                         no_grasps=1,
            #                         grasp_width_img=width_img)
                # fig.savefig('img_result.pdf')
                # time = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
                # fig.savefig('results/{}.img_result.pdf'.format(time))
            grasps = detect_grasps(q_img=q_img,ang_img=ang_img,width_img=width_img,no_grasps=1)
            try:
                col = grasps[0].center[1] + self.img_data.top_left[1] 
                row = grasps[0].center[0] + self.img_data.top_left[0] 
                angle = (grasps[0].angle + 2 * math.pi) % math.pi
                width = grasps[0].width * 2
            except:
                col = self.img_data.top_left[1] 
                row = self.img_data.top_left[0] 
                angle = math.pi
                width = 0
        
        return row, col, angle, width, q_img



#if __name__ == '__main__':
if __name__ == '__main__':
    # Load Network
    path = '/home/zzy/workspace/grasp/gitrepo/pybullet_grasp/grcnn/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    # net = torch.load(path)
    grcnn_inferencer = inferencer(path,use_depth=1,use_rgb=1,input_size=224,save_img = False)


    # Get the compute device


def inference(args_network, color_img,depth_img,args_use_depth=True,args_use_rgb=True, args_n_grasps=1,args_save=True,args_force_cpu=False):
    #args = parse_args()

    # Load image
    rgb=color_img
    depth=np.expand_dims(depth_img, axis=2)
    '''
    logging.info('Loading image...')
    pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(pic)
    pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2)
    '''
    
    # Load Network
    logging.info('Loading model...')
    if args_force_cpu:
        net = torch.load(args_network,map_location='cpu')    # the argument  map_location='cpu'  must be added if u r using a CPU machine
    else:
        net = torch.load(args_network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args_force_cpu)

    img_data = CameraData(width=224,
                          height=224,
                          output_size=224,
                          include_depth=args_use_depth, 
                          include_rgb=args_use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        # post process, gaussian blur and synthesise angle image
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        if args_save:
            grasps=save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args_n_grasps,
                grasp_width_img=width_img
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            grasps=plot_results(fig=fig,
                                 rgb_img=img_data.get_rgb(rgb, False),
                                 grasp_q_img=q_img,
                                 grasp_angle_img=ang_img,
                                 no_grasps=args_n_grasps,
                                 grasp_width_img=width_img)
            #fig.savefig('img_result.pdf')
            time = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
            fig.savefig('results/{}.img_result.pdf'.format(time))
    
    return grasps