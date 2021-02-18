import glob
import cv2
import numpy
import dlib
import numpy as np

from training_data.facepoints_regions import save_cheeks_nose_chin

# change path
absolutePath = '/home/stephen/THESIS_PROTOTYPE_DEV/FACE_SCANNING_TECHNOLOGY_master/'

# pores path
chin_img_path_p = absolutePath + 'training_data/pores/chin_pictures/*'
cheek_img_path_p = absolutePath + 'training_data/pores/cheek_pictures/*'
forehead_img_path_p = absolutePath + 'training_data/pores/forehead_pictures/*'
nose_img_path_p = absolutePath + 'training_data/pores/nose_pictures/*'
eyes_img_path_p = absolutePath + 'training_data/pores/side_eyes/*'

# wrinkles path
chin_img_path_w = absolutePath + 'training_data/wrinkles/chin_pictures/*'
cheek_img_path_w = absolutePath + 'training_data/wrinkles/cheek_pictures/*'
forehead_img_path_w = absolutePath + 'training_data/wrinkles/forehead_pictures/*'
nose_img_path_w = absolutePath + 'training_data/wrinkles/nose_pictures/*'
eyes_img_path_w = absolutePath + 'training_data/wrinkles/side_eyes/*'

# lips path
train_moisturized_lips_path = absolutePath + 'training_data/moisture/train_moisturized_lips/*'
train_dry_lips_path = absolutePath + 'training_data/moisture/train_dry_lips/*'
predictor_path = absolutePath + 'shape_predictor_68_face_landmarks.dat'
roi_path = absolutePath + 'Lips_Moisture_Analysis/lips_analysis/cropped_lips'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
moisturized_lips_list = glob.glob(train_moisturized_lips_path)
dry_lips_list = glob.glob(train_dry_lips_path)

# Txt File path
txt_file_pores = absolutePath + 'training_data/pores/dataset_pores_analyze.txt'
txt_file_wrinkles = absolutePath + 'training_data/wrinkles/dataset_wrinkles_analyze.txt'
txt_file_moisture = absolutePath + 'training_data/moisture/dataset_lips_analyze.txt'


def map_value(x, in_min, in_max, out_min, out_max):
    map_out = int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    if map_out < 0:
        map_out = 0
    elif map_out >= in_max:
        map_out = 100
    return map_out


# PORES FUNCTIONS

# def get_edge_value_pores(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     fm = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return fm

def get_edge_value_pores(image):
    arr = image
    arr_list = arr.tolist()
    r = g = b = 0
    for row in arr_list:
        for item in row:
            r = r + item[2]
            g = g + item[1]
            b = b + item[0]
    total = r + g + b
    red = round(r / total * 100)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 100, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    s1 = 3
    s2 = 20
    xcnts = []

    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            xcnts.append(cnt)

        # printing output
    return len(xcnts) + red


def training_low_high_pores(path, label):
    train_img_list = glob.glob(path)
    edge_values = []
    for train_img in train_img_list:
        img_cv2 = cv2.imread(train_img)
        fm = int(get_edge_value_pores(img_cv2))
        print(label, train_img, fm)
        edge_values.append(fm)
    print(label, edge_values)
    highest_val = max(edge_values)
    lowest_val = min(edge_values)
    f = open(txt_file_pores, "a")
    str_dataset = "{0},{1},{2}\n".format(label, str(lowest_val), str(highest_val))
    f.write(str_dataset)
    f.close()


def get_values_high_low_pores(label=None):
    file1 = open(txt_file_pores, 'r')
    Lines = file1.readlines()

    for line in Lines:
        line_get = line.strip()
        line_list = line_get.split(',')
        if line_list[0] == label:
            low_val = line_list[1]
            high_val = line_list[2]
    return int(low_val), int(high_val)


def train_pores():
    f = open(txt_file_pores, 'w')
    f.close()
    training_low_high_pores(chin_img_path_p, 'chin')
    training_low_high_pores(forehead_img_path_p, 'forehead')
    training_low_high_pores(nose_img_path_p, 'nose')
    training_low_high_pores(cheek_img_path_p, 'cheeks')
    training_low_high_pores(eyes_img_path_p, 'eyes')


# PORES FUNCTIONS

# WRINKLES FUNCTIONS
def get_edge_value_wrinkles(image):
    # Credits to https://stackoverflow.com/users/6230266/shen

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (480, 640), interpolation=cv2.INTER_AREA)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    if lines is not None:
        num_wrinkles = len(lines)
    else:
        num_wrinkles = 0



    return num_wrinkles


def training_low_high_wrinkles(path, label):
    train_img_list = glob.glob(path)
    edge_values = []
    for train_img in train_img_list:
        img_cv2 = cv2.imread(train_img)
        fm = int(get_edge_value_wrinkles(img_cv2))
        print(label, train_img, fm)
        edge_values.append(fm)
    print(label, edge_values)
    highest_val = max(edge_values)
    lowest_val = min(edge_values)
    f = open(txt_file_wrinkles, "a")
    str_dataset = "{0},{1},{2}\n".format(label, str(lowest_val), str(highest_val))
    f.write(str_dataset)
    f.close()


def get_values_high_low_wrinkles(label=None):
    file1 = open(txt_file_wrinkles, 'r')
    Lines = file1.readlines()

    for line in Lines:
        line_get = line.strip()
        line_list = line_get.split(',')
        if line_list[0] == label:
            low_val = line_list[1]
            high_val = line_list[2]
    return int(low_val), int(high_val)


def train_wrinkles():
    f = open(txt_file_wrinkles, 'w')
    f.close()
    training_low_high_wrinkles(chin_img_path_w, 'chin')
    training_low_high_wrinkles(forehead_img_path_w, 'forehead')
    training_low_high_wrinkles(nose_img_path_w, 'nose')
    training_low_high_wrinkles(cheek_img_path_w, 'cheeks')
    training_low_high_wrinkles(eyes_img_path_w, 'eyes')


# WRINKLES FUNCTIONS

# LIPS MOISTURE FUNCTIONS
def find_lips(input_img):
    frame = input_img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            if n == 3:
                lips_origin_y = y
            elif n == 6:
                lips_origin_x = x
            elif n == 10:
                lips_width = x
            elif n == 11:
                lips_height = y

    lips_cropped = frame[lips_origin_y:lips_height, lips_origin_x:lips_width]
    return lips_cropped


def get_edge_value_lips(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def average_color(image):
    avg_color_per_row = numpy.average(image, axis=0)
    avg_color = numpy.average(avg_color_per_row, axis=0)
    return avg_color


def get_features_lips_m():
    # global edge_feature, r_feature, g_feature, b_feature
    for train_moisturized_path in moisturized_lips_list:
        print(train_moisturized_path)

        try:
            frame = cv2.imread(train_moisturized_path)
            roi_lips = find_lips(frame)

            edge_value = get_edge_value_lips(roi_lips)
            # edge_feature.append(edge_value)

            r_value = average_color(roi_lips)[0]
            g_value = average_color(roi_lips)[1]
            b_value = average_color(roi_lips)[2]

            f = open(txt_file_moisture, "a")
            str_dataset = "{0},{1},{2},{3}".format(str(edge_value), str(r_value), str(g_value), str(b_value))
            str_dataset = str_dataset + "," + str(1) + "\n"
            f.write(str_dataset)
            f.close()
        except:

            print("Cant Find face for", train_moisturized_path)
            continue


def get_features_lips_d():
    # global edge_feature, r_feature, g_feature, b_feature
    for train_dry_path in dry_lips_list:
        print(train_dry_path)
        try:
            frame = cv2.imread(train_dry_path)
            roi_lips = find_lips(frame)

            edge_value = get_edge_value_lips(roi_lips)
            # edge_feature.append(edge_value)

            r_value = average_color(roi_lips)[0]
            g_value = average_color(roi_lips)[1]
            b_value = average_color(roi_lips)[2]

            f = open(txt_file_moisture, "a")
            str_dataset = "{0},{1},{2},{3}".format(str(edge_value), str(r_value), str(g_value), str(b_value))
            str_dataset = str_dataset + "," + str(0) + "\n"
            f.write(str_dataset)
            f.close()

        except:
            print("Cant Find face for", train_dry_path)


def train_lips():
    f = open(txt_file_moisture, "w")
    f.close()
    if len(moisturized_lips_list) == len(dry_lips_list):
        get_features_lips_m()
        get_features_lips_d()
    else:
        print('moisturized_folder', len(moisturized_lips_list))
        print('dry_folder', len(dry_lips_list))
        print('two dataset is not balance')


# LIPS MOISTURE FUNCTIONS


if __name__ == "__main__":
    print('-' * 50, 'TRAINING FACIAL PORES', '-' * 50)
    train_pores()
    print('-' * 50, 'TRAINING FACIAL WRINKLES', '-' * 50)
    train_wrinkles()
    print('-' * 50, 'TRAINING LIPS MOISTURE', '-' * 50)
    train_lips()
    # absolutePath = '/home/stephen/THESIS_PROTOTYPE_DEV/FACE_SCANNING_TECHNOLOGY_master/'
    # resolution_x, resolution_y = 600, 600
    # path = 'training_data/pores/cheek_pictures/left_cheek_20210218-163120.jpg'
    #
    # # test Image
    # test_image_path = absolutePath + path
    # input_img = cv2.imread(test_image_path)
    # print(get_edge_value_wrinkles(input_img))
