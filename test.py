#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import sys
sys.path.append('.')
sys.path.append('..')

import os
import time
import numpy as np
import argparse
import cv2
from tqdm import tqdm
from termcolor import cprint
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

from tools import metrics
from models import build_model


# global parameters
input_size = 224
n_classes = 2
crop_image = False
to_tensor = alb.Compose([
    alb.Resize(input_size, input_size),
    alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])


def inference(args):
    net = build_model.build_model(args)
    net.cuda()
    net.load_state_dict(torch.load(args.ckpt), strict=False)
    net = load_dict(args, net)
    net.eval()

    ds = dataloader(args.valid_db)
    print('Start Inference ......')
    sum_time = 0.0
    lines_res = []
    
    with torch.no_grad():
        l = 0
        for img_path, data, label, len_ in ds:
            print("current count: ------- ", l)
            # img = to_tensor(data)                 # for torch transform
            img = to_tensor(image=data)['image']    # for albumentations
            img = torch.unsqueeze(img, 0)
            img = img.type(torch.FloatTensor)
            img = img.cuda()

            time_start = time.time()
            out = net(img)[1]
            out = F.softmax(out, dim=1)
            out = out.squeeze().cpu().numpy()

            line_res = img_path + ' ' + str(label) + ' ' + str(out[-1]) + '\n'
            lines_res.append(line_res)

            time_end = time.time()
            sum_time += float(time_end-time_start)

            print(img_path, out)
            l += 1
    
    avg_time = sum_time / len_
    fps = 1. / avg_time
    print('avg time = %.5f, fps = %.2f'%(avg_time, fps))

    with open(args.out, 'w') as fid:
        fid.writelines(lines_res)
    performance_str = metrics.eval_roc(args.out)



def sampler(img_path, bboxes):
    if crop_image:
        img = get_img_crop(img_path, bboxes)
    else:
        img = cv2.imread(img_path)

    # exchange channel: BGR ==> RGB
    img = img[: ,:, [2, 1, 0]]
    if img is None:
        return None

    img = cv2.resize(img, (input_size, input_size))
    return img



def dataloader(path_input_list):
    with open(path_input_list, 'r') as fid:
        lines = fid.readlines()
    for l, line in enumerate(lines):
        name_splits = line.strip().split()
        img_path = name_splits[0]
        label = int(float(name_splits[1]))
        bboxes = [int(x) for x in name_splits[2:]]
        data = sampler(img_path, bboxes)
        
        if data is None:
            print("image none")
            yield img_path, np.zeros((input_size, input_size,3)), "0", len(lines)
        yield img_path, data, label, len(lines)



def load_dict(args, model, insert_args=None):
    if os.path.isfile(args.ckpt):
        cprint('=> loading pth from {} ...'.format(args.ckpt), 'red')
        checkpoint = torch.load(args.ckpt)
        _state_dict = clean_dict(model, checkpoint['state_dict'], insert_args=insert_args)
        model.load_state_dict(_state_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        print("=> No checkpoint found at '{}'".format(args.ckpt))
    return model



def clean_dict(model, state_dict, insert_args=None):
    _state_dict = OrderedDict()
    print(list(state_dict.items())[0][0])
    print(list(model.state_dict().items())[0][0] )
    # exit(1)
    
    for k, v in state_dict.items():
        k_ = k.split('.')
        assert k_[0] == 'module'
        if insert_args is not None:
            k_.insert(insert_args[0], insert_args[1])
        
        k = '.'.join(k_[1:])

        if k in model.state_dict().keys() and \
           v.size() == model.state_dict()[k].size():
            _state_dict[k] = v
            cprint(' : successfully load {} - {}'.format(k, v.shape), 'green')
        else:
            try:
                _state_dict[k] = model.state_dict()[k]
                cprint(' : ignore {} - {}'.format(k, v.shape), 'yellow')
            except:
                cprint(' : delete {} - {}'.format(k, v.shape), 'yellow')

    for k, v in model.state_dict().items():
        if k in _state_dict.keys():
            continue
        _state_dict[k] = v
    return _state_dict



def get_img_crop(img_path, bbox, scale=2.5):
    """
    crop a raw image from bbox.
    bbox is gotton from a face detection model.
    x_min, y_min, face_x, face_y = bbox
    """
    img = cv2.imread(img_path)

    if img is None:
        print('reading empty image: ', img_path)
        return None

    shape = img.shape
    h, w = shape[:2]
    x_min, y_min, face_x, face_y = bbox
    x_mid = x_min + face_x / 2.0
    y_mid = y_min + face_y / 2.0

    if x_mid<=0 or y_mid<=0 or face_x<=0 or face_y<=0:
        return None

    x_min = int( max(x_mid - face_x * scale / 2.0, 0) )
    x_max = int( min(x_mid + face_x * scale / 2.0, w) )
    y_min = int( max(y_mid - face_y * scale / 2.0, 0) )
    y_max = int( min(y_mid + face_y * scale / 2.0, h) )

    if x_min >= x_max or y_min >= y_max:
        return None   
    return img[y_min:y_max, x_min:x_max, :]



if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('-a', '--arch', metavar='ARCH', default='deit', help='model architecture')
    parse.add_argument('--crop-scale', default=2.5, type=float, help='scale to crop a face from raw image')
    parse.add_argument('--ckpt', dest = 'ckpt', type = str, default='./ckpt/checkpoint_20210524/epoch_4.pth.tar')
    parse.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parse.add_argument('--valid_db', dest = 'valid_db', type = str, default="/home/projects/list/list_72/cbsr_mas_v6_hifi-mask-test_bbox.txt")
    parse.add_argument('--out', dest = 'out', type = str, default="/home/projects/list/test_result_transformer/checkpoint_20210524_epoch_4_cbsr_mas_v6_hifi-mask-test_bbox.txt")
    args = parse.parse_args()
    inference(args)


