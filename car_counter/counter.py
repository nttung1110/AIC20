import numpy as np
import os
import matplotlib.path as mplPath
import cv2
PATH_ROI = "../data/AIC20_track1/ROIs"
PATH_VIDEO = "../data/AIC20_track1/Dataset_A"
PATH_TRACKING = "../Center_net/new_iou_info_tracking"
# PATH_TRACKING = "../Center_net/info_split"
PATH_VIDEO_OUT = "../Center_net/counting_visualized"
VISUALIZED = True

def load_roi():
    roi_list = {}
    for file_name in os.listdir(PATH_ROI):
        if(file_name.endswith(".txt")):
            full_path_name = os.path.join(PATH_ROI, file_name)
            file_p = open(full_path_name, "r")
            content_list = file_p.read().splitlines()
            content_list = [(int(point.split(',')[0]), int(point.split(',')[1])) for point in content_list]
            roi_list[file_name[:-4]] = content_list
    return roi_list

def out_of_roi(center, poly):
    path_array = []
    for poly_point in poly:
        path_array.append([poly_point[0], poly_point[1]])
    path_array = np.asarray(path_array)
    polyPath = mplPath.Path(path_array)

    return polyPath.contains_point(center, radius = 0.5)

def validate_center(center, use_off_set, roi_list):
    const = 5
    off_set = [(const, const), (const, -const), (-const, const), (-const, -const)]

    if not use_off_set:
        return  out_of_roi(center, roi_list)

    for each_off_set in off_set:
        center_change = (center[0] + each_off_set[0], center[1] + each_off_set[1])
        if out_of_roi(center_change, roi_list):
            return False
    return True

def out_of_range_bbox(tracking_info, width, height, off_set):
    x_min = int(tracking_info[4])
    y_min = int(tracking_info[5])
    x_max = int(tracking_info[6])
    y_max = int(tracking_info[7])
    return ((x_min-off_set <=0) or (y_min-off_set<=0) or (x_max+off_set>=width) or (y_max+off_set>=height))


def find_latest_object_and_vote_direction(frame_id_list, cur_fr_id, tracking_info, delta_fix, target_obj_id, roi_list, width, height):
    exist_latest_obj = False
    count_out = 0
    count_in = 0
    for delta in range(1, delta_fix):
        pre_index = np.where(frame_id_list == (cur_fr_id - delta))[0]
        for each_pre_index in pre_index:
            if tracking_info[each_pre_index][3]==target_obj_id:
                exist_latest_obj = True
                pre_obj_center = center_box(tracking_info[each_pre_index][4:])
                if out_of_range_bbox(tracking_info[each_pre_index], width, height, 2):
                    count_out += 1
                else:
                    if validate_center(pre_obj_center, False, roi_list):
                        count_in += 1
                    else:
                        count_out += 1
    return count_out, count_in, exist_latest_obj

def center_box(cur_box):
    return (int((cur_box[0]+cur_box[2])/2), int((cur_box[1]+cur_box[3])/2))

def draw_roi(roi_list, image):
    start_point = roi_list[0]
    for end_point in roi_list[1:]:
        cv2.line(image, start_point, end_point, (0,0,255), 2)
        start_point = end_point
    return image

def car_counting(vid_name, roi_list):
    video_name = vid_name+".mp4"
    print("Processing", vid_name)
    tracking_info = np.load(PATH_TRACKING + '/info_' + video_name + '.npy', allow_pickle = True)
    N = tracking_info.shape[0]
    frame_id = tracking_info[:, 1].astype(np.int).reshape(N)
    delta_fix = 10
    obj_id = tracking_info[:, 3].astype(np.int).reshape(N)
    results = []
    num_car_out = 0

    input = cv2.VideoCapture(PATH_VIDEO + '/' + vid_name + '.mp4')
    width = int(input.get(3)) # get width
    print("Visualizing ", vid_name)
    height = int(input.get(4)) #get height

    num_truck_out = 0
    already_count = []
    for fr_id in range(1, max(frame_id)+1):
        index_cur_fr = np.where(frame_id==fr_id)[0]
        for index_box in index_cur_fr:
            cur_box = tracking_info[index_box][4:]
            cur_center = center_box(cur_box)
            is_inside_roi = validate_center(cur_center, False, roi_list)
            cur_obj_id = tracking_info[index_box][3]
            if not is_inside_roi:# current car is outside roi
                count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height)
                if is_ok: #exist object
                    # pre_obj_center = center_box(latest_obj[4:])
                    # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                    if count_in>=count_out and cur_obj_id not in already_count: # previous car lies inside roi
                        already_count.append(cur_obj_id)
                        if tracking_info[index_box][0] == 1:
                            num_car_out += 1
                            num_object_out = num_car_out
                        else:
                            num_truck_out += 1
                            num_object_out = num_truck_out
                        results.append([fr_id, num_object_out, cur_center[0], cur_center[1]])
            else : # using offset to refine again
                is_out = out_of_range_bbox(tracking_info[index_box], width, height, 2)
                if is_out:
                    count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height)
                    if is_ok: #exist object
                        # pre_obj_center = center_box(latest_obj[4:])
                        # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                        if count_in>=count_out and cur_obj_id not in already_count: # previous car lies inside roi
                            already_count.append(cur_obj_id)
                            if tracking_info[index_box][0] == 1:
                                num_car_out += 1
                                num_object_out = num_car_out
                            else:
                                num_truck_out += 1
                                num_object_out = num_truck_out
                            results.append([fr_id, num_object_out, cur_center[0], cur_center[1]])
    if VISUALIZED:
        output = cv2.VideoWriter(PATH_VIDEO_OUT + '/' + vid_name + '.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 5.0, (width, height))
        idx = 0
        results = np.array(results)
        N = len(results)
        print(N)
        frame_id = results[:, 0].astype(np.int).reshape(N)
        while (input.isOpened()):
            ret, frame = input.read()
            if not ret:
                break
            idx += 1
            indx_cur_fr = np.where(frame_id == idx)[0]
            annotate_fr = draw_roi(roi_list, frame)
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    cv2.putText(annotate_fr, str(count_object).zfill(5), (cur_annotate[2], cur_annotate[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            output.write(annotate_fr)
            if idx == 900:
                break
        input.release()
        output.release()
    return results

if __name__ == "__main__":
    roi_list = load_roi()
    for video_name in os.listdir(PATH_VIDEO):
        if video_name .endswith(".mp4"):
            roi_vid_name = video_name[:-4].split("_")[0]+ "_" + video_name[:-4].split("_")[1]
            results = car_counting(video_name[:-4], roi_list[roi_vid_name])
