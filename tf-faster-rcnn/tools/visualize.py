import cv2
import os
import numpy as np
import time

PATH_VIDEO = '../data/AIC20_track1/Dataset_A'
# PATH_INFO = '../result'
PATH_INFO = '../result'
PATH_SMALL = '../small_info'
PATH_ORI = '../ori_info'
PATH_VIS = '../video_visualize'
THRESH = 0.6

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

def visualize(video, info, writer):
    for i in range(1, 901):
        _, frame = video.read()
        boxes = info[info[:, -1] == i]
        for box in boxes:
            box = box[:4].astype(np.int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite("../test_fr/frames"+str(i)+".jpg", frame)
        writer.write(frame)

def prune(info, thresh=0.6):
    ans = []
    for idx in range(1, 901):
        box = info[info[:,-1] == idx]
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



def process(video_name):
    path_info = '%s/info_%s.mp4.npy' % (PATH_INFO, video_name)
    path_small_info = '%s/info_%s.mp4.npy' % (PATH_SMALL, video_name)
    path_ori_info = '%s/info_%s.mp4.npy' % (PATH_ORI, video_name)
    path_video = '%s/%s.mp4' % (PATH_VIDEO, video_name)
    if not os.path.exists(PATH_VIS):
        os.mkdir(PATH_VIS)
    path_vis = '%s/%s.avi' % (PATH_VIS, video_name)

    #check exist annotated video frame
    if os.path.isfile(path_info)==False:
        return
    if os.path.isfile(path_small_info)==False:
        return
    if os.path.isfile(path_ori_info)==False:
        return

    info = np.load(path_info)
    small_info = np.load(path_small_info)
    ori_info = np.load(path_ori_info)
    print(info.shape)
    print(small_info.shape)
    print(ori_info.shape)


    video = cv2.VideoCapture(path_video)
    width = int(video.get(3)) # get width
    height = int(video.get(4)) #get height
    print("Video shape:(%d,%d)"%(width, height))
    writer = cv2.VideoWriter(path_vis, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))
    res = []
    bbox1 = []
    bbox2 = []
    bbox3 = []
    for idx in range(1, 901):
        if(info.shape[0]!=0):
            bbox1 = info[info[:, -1]==idx]
        if(ori_info.shape[0]!=0):
            bbox2 = ori_info[ori_info[:, -1] == idx]
        if(small_info.shape[0]!=0):
            bbox3 = small_info[small_info[:, -1] == idx]
        for box in bbox1:
            # coor = box[-4:].astype(np.int)
            # res.append(box[:2].tolist() + box[-4:].tolist())
            # assert len(res[-1]) == 6
            coor = box[:4].astype(np.int)
            # res.append([box[-1], box[-2]] + box[:4].tolist())
            res.append(box[:4].tolist()+ [box[-2], box[-1]])
            assert len(res[-1]) == 6
        for box in bbox2:
            coor = box[:4].astype(np.int)
            # res.append([box[-1], box[-2]] + box[:4].tolist())
            # res.append([box[-1], box[-2]] + box[:4].tolist())
            res.append(box[:4].tolist()+ [box[-2], box[-1]])
            assert len(res[-1]) == 6

        for box in bbox3:
            box[:4] = box[:4] / 3
            box[0] = box[0] + 480
            box[2] = box[2] + 480
            coor = box[:4].astype(np.int)
            # res.append([box[-1], box[-2]] + box[:4].tolist())
            res.append(box[:4].tolist()+ [box[-2], box[-1]])
            assert len(res[-1]) == 6

    res = prune(np.array(res))
    np.save('../info/info_%s.mp4' % video_name, res)
    # visualize(video, np.asarray(res), writer)
    # video.release()
    # writer.release()

    return np.array(res)

if __name__ == '__main__':
    list_file = os.listdir(PATH_VIDEO)
    video_names = [video_name.split('.')[0] for video_name in list_file if(video_name.endswith(".mp4"))]
    video_names.sort()
    # remained_vid = ["cam_5", "cam_5_rain", "cam_7_rain", "cam_13"]
    for video_name in video_names:
        #print('Process video %s.mp4' % video_name)
        # if(video_name in remained_vid):
        duration = time.time()
        print(video_name)
        process(video_name)
        duration = time.time() - duration
        #print('Process video %s.mp4 take %f second' % (video_name, duration))