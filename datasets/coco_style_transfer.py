"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torchvision
from torchvision import transforms as Tr

import glob
from PIL import Image
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

class CocoStyleTransfer(torchvision.datasets.CocoDetection):
    def __init__(self,content_folder, style_folder,img_size):
        # super(CocoStyleTransfer, self).__init__(coco_img_folder, coco_ann_file)
        style_images = glob.glob(str(style_folder) + '/*')
        style_images = [os.path.normpath(path) for path in style_images]
#         self._transforms = transforms
        self.std = std
        self.mean = mean
        self._transforms = Tr.Compose([
            Tr.ToTensor(),
            Tr.Normalize(self.mean, self.std)
        ])
        content_images = glob.glob(str(content_folder) + '/*')
        content_images = [os.path.normpath(path) for path in content_images]
        
        print("len(content_images),len(style_images):",len(content_images),len(style_images))
        self.images_pairs = [[x,y] for x in content_images for y in style_images ] 
        self.img_size=img_size
    def center_crop(self,img,max_img_size=600):
        width, height = img.size   # Get dimensions
        if width>=max_img_size or height>=max_img_size:
            new_width=max_img_size
            new_height=int(max_img_size/width*height)
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            img = img.crop((left, top, right, bottom))
        return img
    
    def img_resize(self,im,div=32):
        desired_size=self.img_size
        h, w = im.size
        if h>w:
            new_h=desired_size
            new_w=int(new_h/h*w)
        else:
            new_w=desired_size
            new_h=int(new_w/w*h)
            
        
        new_w = (new_w%div==0) and new_w or (new_w + (div-(new_w%div)))
        new_h = (new_h%div==0) and new_h or (new_h + (div-(new_h%div)))
            
        new_im  = im.resize((new_h,new_w), Image.LANCZOS).convert("RGB")
        
        noise_new_im=np.array(new_im)
#         sigma=3
#         noise_new_im+=+np.random.randn(new_w,new_h,3) * sigma / 255
        return noise_new_im
        
    def image2div(self,img,div=32):
        width, height = img.size
        nw = (width%div==0) and width or (width + (div-(width%div)))
        nh = (height%div==0) and height or (height + (div-(height%div)))
        img  = img.resize((nw,nh), Image.LANCZOS).convert("RGB")
        
        return img
        
    def __getitem__(self, pair_idx):
        content_image_path, style_image_path = self.images_pairs[pair_idx]
        c_name=os.path.basename(content_image_path).split(".")[0]
        s_name=os.path.basename(style_image_path).split(".")[0]
        print(content_image_path,style_image_path)
        target = {'content_image_name': c_name, 'style_image_name': s_name}
        print(content_image_path,style_image_path)
        img = Image.open(content_image_path).convert("RGB")
        style_image = Image.open(style_image_path).convert("RGB")
        style_image = self.img_resize(style_image)
        img = self.img_resize(img)
        
        if self._transforms is not None:
            style_image = self._transforms(style_image)
            img = self._transforms(img)
        return img, style_image, target
    
    def __len__(self):
        return len(self.images_pairs)


