import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool import utils 
from typing import List
import torch.autograd.profiler as profiler

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs:List[List[torch.Tensor]]):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []
    print(boxes_and_confs[0][0].shape)
    print(boxes_and_confs[0][1].shape)
    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    
    t1 = time.time()

    #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    output = model(img)
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    #print(torch.cuda.memory_allocated())
    #print(torch.cuda.memory_cached())
    t2 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')
    out0 = output[0].cpu().detach().numpy()
    out1 = output[1].cpu().detach().numpy()

    output = [out0, out1]
    cdevice = torch.device('cpu')
    img = img.to(cdevice)
    torch.cuda.empty_cache()
    return utils.post_processing(conf_thresh, nms_thresh, out0, out1)

