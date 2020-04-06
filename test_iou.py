import numpy as np
import pickle
import os
import json
import time
from iou_tracker.viou_tracker import *
from iou_tracker.iou_tracker import *
path_bbox = "./Center_net/new_bboxes/bboxes"
path_video = "./data/AIC20_track1/Dataset_A"

path_bbox_dla = "./dla_backbone/json_frames_dla_34"

path_bbox_resnet = "./resnet50_backbone/json_frames_res_50"

def format_bbox_centernet(video_name, file_name):

    file_content = open(os.path.join(path_bbox,file_name),'rb')
    content = pickle.load(file_content)
    print("Processing:",video_name)
    data = []

    for fr_id, fr_content in enumerate(content):
        dets = []
        c_bboxes = fr_content[1]
        t_bboxes = fr_content[2]
        for bb in c_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 1})
        for bb in t_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 2})
        data.append(dets)
    file_content.close()
    return data

def run_tracking_bbox_dla():
    for file_name in os.listdir(path_bbox_dla):
        vid_name = file_name[:-5]
        file_content = open(os.path.join(path_bbox_dla,file_name),'rb')
        content = json.load(file_content)
        print("Processing:",vid_name)
        data = []
        old_frame = 0
        dets = []
        for box in content:
            fr_id = int(box['image_name'].split("_")[-1][:-4])
            print(fr_id)
            box_coor = box["bbox"]
            if fr_id == old_frame:
                dets.append({'bbox': (int(box_coor[0]), int(box_coor[1]), int(box_coor[0] + box_coor[2]), int(box_coor[1]+ box_coor[3])), 'score': box["score"], 'class': box["category_id"]})
            else:
                data.append(dets)
                dets = []
                old_frame = fr_id
                dets.append({'bbox': (int(box_coor[0]), int(box_coor[1]), int(box_coor[0] + box_coor[2]), int(box_coor[1]+ box_coor[3])), 'score': box["score"], 'class': box["category_id"]})
        data.append(dets)
        file_content.close()

        content_video_path = os.path.join(path_video, vid_name+".mp4")
        # results = track_viou_edited(content_video_path, data, 0.3, 0.7, 0.6, 13, 6, "KCF", 1.0)
        results = track_iou_edited(vid_name, data, 0.3, 0.7, 0.3, 10)
        break
        
    return data

def format_bbox_resnet():
    file_content = open(os.path.join(path_bbox,file_name),'rb')
    content = pickle.load(file_content)
    print("Processing:",video_name)
    data = []

    for fr_id, fr_content in enumerate(content):
        dets = []
        c_bboxes = fr_content[1]
        t_bboxes = fr_content[2]
        for bb in c_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 1})
        for bb in t_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 2})
        data.append(dets)
    file_content.close()
    return data

if __name__ == "__main__":
    for file_name in os.listdir(path_bbox):
        vid_name = file_name[:-8]
        # if vid_name == "cam_17":
        data = format_bbox_centernet(vid_name, file_name)
        content_video_path = os.path.join(path_video, vid_name+".mp4")
        # results = track_viou_edited(content_video_path, data, 0.3, 0.7, 0.6, 13, 6, "KCF", 1.0)
        results = track_iou_edited(vid_name, data, 0.3, 0.7, 0.3, 10)
    # run_tracking_bbox_dla()
