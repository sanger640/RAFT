import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

#use python demo.py --model=models/raft-kitti.pth --path=/road/rgb-images/; to run inference on the entire road dataset and save
#the optical representation in road/opf/

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, counter, save_dir):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=1)

    # plt.imshow(img_flo / 255.0)
    # plt.show()
    #print('test1')
    #plt.imsave(f'/road/opf+rgb/2014-06-25-16-45-34_stereo_centre_02/{counter}.png', img_flo[:, :, [2,1,0]])
    number_str = str(counter)
    cv2.imwrite( save_dir + f'/{number_str.zfill(5)}.png', flo[:, :, [2,1,0]])
    #print('test2')
    #cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for fold in glob.glob(os.path.join(args.path, '*')):
            images = glob.glob(fold + '/*.jpg') 
            x = os.path.basename(fold)
            images = sorted(images)
            counter = 1
            save_dir = '/road/opf/'+ x
            os.mkdir(save_dir)
            for imfile1, imfile2 in zip(images[:-1], images[1:]):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                #print(imfile1, imfile2, 'images')
                #print(image1.shape, 'imsize_pre')
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                #print(image1.shape, 'imsize_post')
                flow_low, flow_up = model(image1, image2, iters=10, test_mode=True)
                #print(flow_up.shape, 'flow_size')
                viz(image1, flow_up, counter, save_dir)
                counter+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
