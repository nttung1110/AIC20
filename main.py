import numpy as np
# import scipy
import cv2 
import os
# import seaborn as sns
# import tensorflow as tf
def vid2frames(video_path, ratio, frames_path):
    count = 0
    for video_name in os.listdir(video_path):
        if(video_name=="cam_3.mp4"):#if(video_name.endswith(".mp4")):
            count+=1
            print("Processing video ",count)
            cap = cv2.VideoCapture(os.path.join(video_path, video_name))
            i = 1
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if i % ratio == 0:
                    name = video_name+"_fr"+str(i)+".jpg"
                    cv2.imwrite(os.path.join(frames_path, name), frame)
                i+=1
                if(i==200):
                    break
            cap.release()
            cv2.destroyAllWindows()
            if(count == 2):
                break
            
if __name__ == "__main__":
    vid_path = "./data/AIC20_track1/Dataset_A"
    frames_path = "./sampled_frames"
    ratio = 10
    vid2frames(vid_path, ratio, frames_path)