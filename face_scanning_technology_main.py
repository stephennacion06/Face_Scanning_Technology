from face_scan_modules import pores_analysis_process, wrinkle_analysis_process, \
    lips_analysis_process
from training_data.facepoints_regions import save_forehead, save_eye_left_right, \
    save_cheeks_nose_chin, find_lips, save_regions, save_time_regions, regions3_display, \
    regions3_display_label

from db_facescan import insert_db

import cv2
import time
import glob

# path to change
absolutePath = '/home/stephen/THESIS_PROTOTYPE_DEV/FACE_SCANNING_TECHNOLOGY_master/'
resolution_x, resolution_y = 600, 600

test_path = absolutePath + 'testing_pictures/*'
test_img_list = glob.glob(test_path)

for test_image_path in test_img_list:
    # test Image
    input_img = cv2.imread(test_image_path)
    input_img = cv2.resize(input_img, (resolution_x, resolution_y), interpolation=cv2.INTER_AREA)

    regions_dict = {'forehead': save_forehead(input_img), 'side_eyes': save_eye_left_right(input_img),
                    'front_face': save_cheeks_nose_chin(input_img), 'lips': find_lips(input_img)}


    time_now = time.strftime("%Y%m%d-%H%M%S")
    save_regions(regions_dict)
    save_time_regions(regions_dict, time_now)
    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pores_dict = pores_analysis_process(regions_dict)
    wrinkles_dict = wrinkle_analysis_process(regions_dict)
    moisture_dict = lips_analysis_process(regions_dict)
    # regions3_display(input_img, regions_dict)
    regions3_display_label(input_img, regions_dict, pores_dict, wrinkles_dict, moisture_dict)
    insert_db(date_time, pores_dict, wrinkles_dict, moisture_dict)
