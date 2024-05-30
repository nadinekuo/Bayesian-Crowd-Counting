from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
import xml.etree.ElementTree as ET


# Function to calculate the new size of an image while maintaining its aspect ratio
def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:                 # Constrain the shortest img side (h or w)
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h   # e.g., if im_h=400, min_size=512, ratio=1.28
            im_h = min_size
            im_w = round(im_w*ratio)        # e.g., if im_w=600, new im_w=round(600*1.28)=768
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size         # In this case, h is the shortest side constrained to max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio   # Returns the new height, width, and the resizing ratio

# Function to find the average distance to the nearest three annotation points
# points = array of shape (h, w, 3) if RGB
def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

# Extract <point> tags from annotation XML and creates np matrix of coords
def extract_xml_points(xml_ann_path):
    tree = ET.parse(xml_ann_path)
    root = tree.getroot()
    points = []
    for obj in root.findall('object'):
        for point in obj.findall('point'):
            x = int(point.find('x').text)
            y = int(point.find('y').text)
            points.append([x, y])
    points_arr = np.array(points)
    return points_arr

def generate_data(im_path):  # /UCF-QNRF_ECCV18/Train/img_XXXX.jpg
    im = Image.open(im_path)   # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=4256x2832 at 0x125CDA190>
    im_w, im_h = im.size
    ann_path = im_path.replace('.jpg', '.xml')   # /UCF-QNRF_ECCV18/Train/img_XXXX_ann.mat
    # Parse xml and create 2D np array of all annotation points
    points = extract_xml_points(ann_path)
    # Filter points within the image boundaries
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    # Get new h and w after resizing img based on min and max size, and the resize ratio used
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)  # min_size: 512, max_size: 2048
    im = np.array(im)  # Array of shape (h, w, 3) if RGB
    if rr != 1.0:     # If img size changed
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr              # Scale points according to the resizing ratio
    # Convert array back to <PIL.Image.Image image mode=RGB size=4256x2832 at 0x127D8D810>
    return Image.fromarray(im), points    # Return the resized image and the adjusted points from ann.mat


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='/CVPR2023',
                        help='original data directory')
    parser.add_argument('--data-dir', default='/CVPR2023-Train-Val-Test',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048

    for phase in ['Train', 'Test']:
        sub_dir = os.path.join(args.origin_dir, phase)   # /CVPR2023/Train e.g.
        if phase == 'Train2':
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(save_dir, sub_phase)  # /CVPR2023-Train-Val-Test/train e.g. 
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                with open('CVPR2023/{}-test.txt'.format(sub_phase)) as f:   # train.txt or val.txt containing rows of img_XXXX.jpg
                    for i in f:
                        im_path = os.path.join(sub_dir, i.strip())  # /UCF-QNRF_ECCV18/Train/img_XXXX.jpg
                        name = os.path.basename(im_path)
                        print(name)                                 # Tail: img_XXXX.jpg
                        # Get rezized im and points
                        # im = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=4256x2832 at 0x1049D8B10>
                        # points = adjusted annotation points
                        im, points = generate_data(im_path)         
                        if sub_phase == 'train':  # For train set, we also append the avg distance to nearest 3 ann points
                            dis = find_dis(points)
                            points = np.concatenate((points, dis), axis=1)
                        im_save_path = os.path.join(sub_save_dir, name)   # /UCF-Train-Val-Test/train/img_XXXX.jpg
                        im.save(im_save_path)
                        # [[ 151.4106   1876.5825    276.90872 ] <-- .npy contents for train
                        # [ 183.36658  1620.9347    139.4184  ]
                        # ....  ]
                        gd_save_path = im_save_path.replace('jpg', 'npy')  # img_XXXX.npy stores the adjusted points and dist
                        np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')  # /UCF-Train-Val-Test/test
            if not os.path.exists(sub_save_dir):           
                os.makedirs(sub_save_dir)
            im_list = glob(os.path.join(sub_dir, '*jpg'))  # Get all .jpg files from /UCF-QNRF_ECCV18/Test
            for im_path in im_list:
                name = os.path.basename(im_path)           # Tail: img_XXXX.jpg 
                print(name)
                im, points = generate_data(im_path)        # Also for test imgs, rezize img and adjust ann points
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                # [[ 445.19424 1790.518  ]  <-- .py contents for test: adjusted ann points
                #  [ 474.8705  1739.259  ]
                #        ... ] 
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)
