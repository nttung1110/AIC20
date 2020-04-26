
import numpy as np
import json
import os
PATH_COUNTING_RESULTS = "../dla_34_new_track_new_count/results_lam_tung"
out_csv_file = "./result_submissions/dla_34_new_track_new_count/submissions_dla_tung_lam_final.csv"
def build_mapping_dictionary():
    file_id = open(PATH_ID_LIST, "r")
    file_list = file_id.read().splitlines()
    my_dict = {}
    for each in file_list:
        each_id = each.split(" ")[0]
        each_filename = each.split(" ")[1][:-4]
        my_dict[each_filename] = each_id
    return my_dict

def write_submission():
    csv_file = open(out_csv_file, "w+")
    csv_file.write(','.join(['video_clip_id', 'frame_id', 'movement_id', 'vehicle_class_id']))
    csv_file.write('\n')
    count = 0
    for file_results in os.listdir(PATH_COUNTING_RESULTS):
        print("Num file processing:", count)
        file_p = open(os.path.join(PATH_COUNTING_RESULTS, file_results), "r")
        contents = file_p.read().splitlines()
        for line in contents:
            out = line.split(",")
            csv_file.write(line)
            csv_file.write('\n')
        file_p.close()
        count += 1

    csv_file.close()


if __name__ == "__main__":
    write_submission()