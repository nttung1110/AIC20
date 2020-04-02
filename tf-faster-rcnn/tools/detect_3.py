#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.resnet_v1 import resnetv1

CLASSES_1 = ('__background__',  # always index 0
                     'dog', 'person', 'cat', 
                     'tv', 'car', 'meatballs', 
                     'marinara sauce', 'tomato soup', 'chicken noodle soup',
                     'french onion soup', 'chicken breast', 'ribs', 
                     'pulled pork', 'hamburger', 'cavity')
CLASSES_2 = ('__background__', 'car')

NETS = {
    'res101': ('res101_faster_rcnn_iter_10000.ckpt',),}

DATASETS= {
    'car_track1': ('car_track1_train',),
    'tiny_car_track1':('tiny_car_track1_train',),
    'horizontal_car_track1': ('horizontal_car_track1_train',)}

VIDEO_DIR = '../data/AIC20_track1/Dataset_A'
# OUTPUT_DIR = '../output_bbox'
OUTPUT_DIR = '../output_bbox_3'

def store_bbox(im, class_name, dets, info, idx, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        info.append(np.append(dets[i], idx))

def extract_bbox(sess, net, im, idx, info, CLASSES):
    scores, boxes = im_detect(sess, net, im)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        if CLASSES[cls_ind] == "car":
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32)

            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            store_bbox(im, cls, dets, info, idx, thresh=CONF_THRESH)

def overlap(anchor, other):
    dx = min(anchor[2], other[2]) - max(anchor[0], other[0])
    dy = min(anchor[3], other[3]) - max(anchor[1], other[1])
    area = 0.0
    if (dx > 0) and (dy > 0):
        area = dx * dy
    anchor_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    overlap_rate = area / max(anchor_area, other_area)
    return overlap_rate

def prune(info, thresh=0.6):
    ans = []
    for idx in range(1, 1801):
        print(idx)
        box = info[info[:,5] == idx]
        N = box.shape[0]
        mark = -1.0 * np.ones((N, ))
        count = 0
        for i in range(N):
            if (mark[i] != -1):
                continue
            for j in range(i, N):
                rate = overlap(box[i,:4], box[j, :4])
                if rate >= thresh:
                    mark[j] = count
            count = count + 1

        for i in range(count):
            newbox = []
            b = box[mark == i]
            newbox.append(np.mean(b[:,0]))
            newbox.append(np.mean(b[:,1]))
            newbox.append(np.mean(b[:,2]))
            newbox.append(np.mean(b[:,3]))
            newbox.append(np.mean(b[:,4]))
            newbox.append(idx)
            ans.append(newbox)

    return np.array(ans)

def save_info_list(info, video_name, model_type):
    # PATH_INFO = '../result'
    # PATH_SMALL = '../small_info'
    # PATH_ORI = '../ori_info'
    print('Saving %s info' % video_name)
    path = ['../result', '../small_info', '../ori_info']
    path_type = path[model_type]
    if not os.path.exists(path_type):
        os.mkdir(path_type)

    vid_name = "info_"+video_name+".npy"
    np.save(os.path.join(path_type, vid_name), info)

def merge_bbox(info, smal, hori):
    smal[:,:4] = smal[:,:4] / 3
    smal[:,0] = smal[:,0] + 480
    smal[:,2] = smal[:,2] + 480
    res = np.vstack([info, smal, hori])
    return prune(res)

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    info_list = [[],[],[]]

    # model path
    demonet = 'res101'
    datasets = ['car_track1', 'tiny_car_track1', 'horizontal_car_track1']


    for i in range(0, 1): # split sessions here
        print(i)
        tf.reset_default_graph()

        if i == 0:
            CLASSES = CLASSES_1
        else:
            CLASSES = CLASSES_2

        dataset = datasets[i]
        tfmodel = os.path.join(
            'output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                'our server and place them properly?').format(tfmodel + '.meta'))

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True

        # init session
        sess = tf.Session(config=tfconfig)

        # load network
        if demonet == 'res101':
            net = resnetv1(num_layers=101)
        else:
            raise NotImplementedError

        net.create_architecture("TEST", len(CLASSES),
            tag='default', anchor_scales=[4, 8, 16, 32])

        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)

        print('Loaded network {:s}'.format(tfmodel))

        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        video_names = os.listdir(VIDEO_DIR)

        for video_name in video_names:
            # if (video_name.endswith(".mp4")):
            if (video_name == "cam_20.mp4"):

                # read video
                video_path = os.path.join(VIDEO_DIR, video_name)
                video = cv2.VideoCapture(video_path)

                info = []
                idx = 1
                while video.isOpened():
                    print(str(idx)+" "+video_name)
                    isReadable, frame = video.read()

                    if not isReadable:
                        break

                    if i == 1:
                        frame = frame[: 540, 480: 1440,:]
                        sz = frame.shape
                        frame = cv2.resize(frame, (sz[1]* 3, sz[0] * 3))
                    extract_bbox(sess, net, frame, idx, info, CLASSES)
                    idx += 1
                    if(idx==900):
                        break
                    print(np.asarray(info).shape)
                save_info_list(info, video_name, i)
                # info_list[i].append(info)
    # for i in range(6):
    #     info = np.array(info_list[0][i])
    #     smal = np.array(info_list[1][i])
    #     hori = np.array(info_list[2][i])

    #     res = merge_bbox(info, smal, hori) change

    #     np.save('Loc3_%d.npy' % (i + 1), res)
    #     break