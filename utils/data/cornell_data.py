import glob
import os

from utils.dataset_processing import grasp, image
import torch
import numpy as np
import random

class CornellDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path, alfa=1, ds_rotate = 0.0, output_size=224,include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False,use_gauss_kernel = 0.0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param output_size: Size of Central Crop transformation
        :param resize_size: Size of Resize transformation
        """
        super(CornellDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        # artificially increase the number of samples by passing the length directly to the dataset 
        self.len = alfa*self.length
        self.use_gauss_kernel = use_gauss_kernel
        
        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]
        
        self.output_size = output_size
        self.resize_size = output_size
        #self.netinput = netinput
        
    def __getitem__(self, idx):
         # modulo operation to repeatedly sample from the data.
        index = idx % self.length
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(index, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(index, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(index, rot, zoom_factor)

        if self.use_gauss_kernel !=0.0:
            # print('gauss')
            pos_img, ang_img, width_img = bbs.draw_gauss(shape = (self.output_size, self.output_size),use_gauss_kernel = self.use_gauss_kernel)
        else: # use binary map
            # print('bina')
            pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), index, rot, zoom_factor

    def __len__(self):
        return self.len
    
    def _get_crop_attrs(self, idx):
        # load the corresponding ground truth bounding boxes
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        # obtaining the correct attributes in order to rotate correctly around the bounding boxes
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top
    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        # offset to correct the effect of central crop 
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        # gtbbs.resize(self.output_size,self.resize_size)
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.resize_size, self.resize_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.resize_size, self.resize_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    
    def get_rgd(self, idx, rot=0, zoom=1.0):
        
        depth_img = self.get_depth(idx,rot,zoom)
        rgb_img = self.get_rgb(idx,rot,zoom)
        # substitute the BLUE channel with depth
        rgb_img[2,:,:] = depth_img      
        return rgb_img

    def get_rgbd(self, idx, rot=0, zoom=1.0):

        depth_img = self.get_depth(idx,rot,zoom)
        rgb_img = self.get_rgb(idx,rot,zoom)
        # substitute the BLUE channel with depth

        rgbd_img = np.pad(rgb_img,((0,1),(0,0),(0,0)),'constant')
        rgbd_img[3,:,:] = depth_img
      
        return rgbd_img
        