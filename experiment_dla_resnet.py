import json
import numpy as np
import os
import time
import cv2
bbox_dla_path = "./dla_backbone/json_frames_dla_34"
bbox_resnet_path = "./resnet50_backbone/json_frames_res_50"
PATH_VID = "./data/AIC20_track1/Dataset_A"
format_bbox_path = "./dla_backbone/info_split/"
THRESH = 0.3
def check_reidentify_info():
    for file_name in os.listdir(PATH_RESULT):
        print(file_name)
        bbox = np.load(os.path.join(PATH_RESULT, file_name))
        print(bbox[3])
        print(bbox[4])
        break
def read_bbox(bbox_path):
    for vid_name in os.listdir(bbox__path):

        file_content = open(os.path.join(bbox_path,vid_name),'rb')
        bbox = json.load(file_content)
        print(bbox[0].keys())
        LENGTH_FRAME = len(bbox)
        duration = time.time()
        return
        video_name = vid_name[:-8]
        input = cv2.VideoCapture(PATH_VID + '/' + video_name + '.mp4')
        print(PATH_VID + '/' + video_name + '.mp4')
        width = int(input.get(3)) # get width
        print("Processed", video_name)
        height = int(input.get(4)) #get height
        output = cv2.VideoWriter(out_visualized_path + '/' + video_name + '.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (width, height))
        idx = 0

        while (input.isOpened()):
            ret, frame = input.read()
            print(idx)
            if not ret:
                break
            car_and_truck_bboxes = np.concatenate((bbox[idx][1], bbox[idx][2]), axis=0)
            for c_t_box in car_and_truck_bboxes:
                if c_t_box[-1] < THRESH:
                    continue
                c_t_box = c_t_box.astype(np.int)
                cv2.rectangle(frame, (c_t_box[0], c_t_box[1]), (c_t_box[2], c_t_box[3]), (0, 0, 255), 3 )
            output.write(frame)
            # cv2.imwrite(os.path.join(PATH_SFRA,video_name+str(idx)+'.jpg'), frame)
            idx= idx + 1
            if(idx==1000):
                break

        input.release()
        output.release()
        duration = time.time() - duration
        print ('Finish Writing Video takes %f second' % duration)


        file_content.close()
        break

def format_bbox():
    '''
        receive each file as list of frame containing bbox stored in dictionary
        (key 3 and 8 are car and truck)'''
    for vid_name in os.listdir(bbox_path):
        print("Processing video", vid_name)
        info_list = []
        file_content = open(os.path.join(bbox_path,vid_name),'rb')
        bbox = pickle.load(file_content)
        for fr_id, fr_content in enumerate(bbox):
            car_and_truck_bboxes = np.concatenate((fr_content[1], fr_content[2]), axis=0)
            for c_t_box in car_and_truck_bboxes:
                if(c_t_box[-1]< THRESH):
                    continue
                info_list.append(c_t_box[:4].tolist()+[c_t_box[-1], fr_id+1]) #index start from 1
        info_list = np.asarray(info_list)
        np.save('./Center_net/info_new/info_%s.mp4' % vid_name[:-8], info_list)
        file_content.close()

def format_bbox_split(num_skip, bbox_path):
    '''
        receive each file as list of frame containing bbox stored in dictionary
        (key 3 and 8 are car and truck)'''
    for vid_name in os.listdir(bbox_path):
        print("Processing video", vid_name)
        info_list = []
        file_content = open(os.path.join(bbox_path,vid_name),'rb')
        bbox = json.load(file_content)
        frame_set = set()
        for box in bbox:
            # car_and_truck_bboxes = np.concatenate((fr_content[1], fr_content[2]), axis=0)
            fr_id = int(box['image_name'].split("_")[-1][:-4])
            frame_set.add(box['image_name'])
            if fr_id%num_skip == 0:
                if box["score"] >= THRESH:
                    info_list.append(box["bbox"]+[box["score"], fr_id+1, box["category_id"]]) #car

        print(frame_set)
        info_list = np.asarray(info_list)
        np.save(format_bbox_path + 'info_%s.mp4' % vid_name[:-4], info_list)
        file_content.close()
        break

if __name__ == "__main__":
    format_bbox_split(1, bbox_dla_path)