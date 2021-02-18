from train_face_regions import get_edge_value_pores, get_values_high_low_pores, get_edge_value_wrinkles, \
    get_values_high_low_wrinkles, map_value, txt_file_moisture, get_edge_value_lips, average_color

from sklearn.naive_bayes import GaussianNB
import cv2

def pores_analysis_process(reg_dict):
    left_cheek_cropped = reg_dict['front_face']['cheek_left']['roi_image']
    right_cheek_cropped = reg_dict['front_face']['cheek_right']['roi_image']
    nose_cropped = reg_dict['front_face']['nose']['roi_image']
    chin_cropped = reg_dict['front_face']['chin']['roi_image']
    forehead_cropped = reg_dict['forehead']['roi_image']
    eye_left_cropped = reg_dict['side_eyes']['eye_left']['roi_image']
    eye_right_cropped = reg_dict['side_eyes']['eye_right']['roi_image']

    forehead_val = get_edge_value_pores(forehead_cropped)
    nose_val = get_edge_value_pores(nose_cropped)
    left_cheek_val = get_edge_value_pores(left_cheek_cropped)
    right_cheek_val = get_edge_value_pores(right_cheek_cropped)
    chin_val = get_edge_value_pores(chin_cropped)
    eyes_left_val = get_edge_value_pores(eye_left_cropped)
    eyes_right_val = get_edge_value_pores(eye_right_cropped)

    chin_min, chin_max = get_values_high_low_pores('chin')
    forehead_min, forehead_max = get_values_high_low_pores('forehead')
    cheek_min, cheek_max = get_values_high_low_pores('cheeks')
    nose_min, nose_max = get_values_high_low_pores('nose')
    eyes_min, eyes_max = get_values_high_low_pores('eyes')

    forehead_per = map_value(forehead_val, forehead_min, forehead_max, 0, 100)
    nose_per = map_value(nose_val, nose_min, nose_max, 0, 100)
    cheek_left_per = map_value(left_cheek_val, cheek_min, cheek_max, 0, 100)
    cheek_right_per = map_value(right_cheek_val, cheek_min, cheek_max, 0, 100)
    chin_per = map_value(chin_val, chin_min, chin_max, 0, 100)
    left_eye_per = map_value(eyes_left_val, eyes_min, eyes_max, 0, 100)
    right_eye_per = map_value(eyes_right_val, eyes_min, eyes_max, 0, 100)

    upper = round((forehead_per + right_eye_per + left_eye_per)/ 3)
    middle = round((cheek_left_per + cheek_right_per + nose_per)/ 3)
    lower = round(chin_per)
    str_print = "Upper Region: {}%\nMiddle Region: {}%\nLower Region: {}%".format(upper, middle, lower)
    print('-' * 20, 'PORES ANALYSIS', '-' * 40)
    print(str_print)
    print('-' * 77)
    return dict(pores=dict(upper_part=upper, middle_part=middle, lower_part=lower))


def wrinkle_analysis_process(reg_dict):
    left_cheek_cropped = reg_dict['front_face']['cheek_left']['roi_image']
    right_cheek_cropped = reg_dict['front_face']['cheek_right']['roi_image']
    nose_cropped = reg_dict['front_face']['nose']['roi_image']
    chin_cropped = reg_dict['front_face']['chin']['roi_image']
    forehead_cropped = reg_dict['forehead']['roi_image']
    eye_left_cropped = reg_dict['side_eyes']['eye_left']['roi_image']
    eye_right_cropped = reg_dict['side_eyes']['eye_right']['roi_image']

    forehead_val = get_edge_value_wrinkles(forehead_cropped)
    nose_val = get_edge_value_wrinkles(nose_cropped)
    left_cheek_val = get_edge_value_wrinkles(left_cheek_cropped)
    right_cheek_val = get_edge_value_wrinkles(right_cheek_cropped)
    chin_val = get_edge_value_wrinkles(chin_cropped)
    eyes_left_val = get_edge_value_wrinkles(eye_left_cropped)
    eyes_right_val = get_edge_value_wrinkles(eye_right_cropped)

    chin_min, chin_max = get_values_high_low_wrinkles('chin')
    forehead_min, forehead_max = get_values_high_low_wrinkles('forehead')
    cheek_min, cheek_max = get_values_high_low_wrinkles('cheeks')
    nose_min, nose_max = get_values_high_low_wrinkles('nose')
    eyes_min, eyes_max = get_values_high_low_wrinkles('eyes')

    forehead_per = map_value(forehead_val, forehead_min, forehead_max, 0, 100)
    nose_per = map_value(nose_val, nose_min, nose_max, 0, 100)
    cheek_left_per = map_value(left_cheek_val, cheek_min, cheek_max, 0, 100)
    cheek_right_per = map_value(right_cheek_val, cheek_min, cheek_max, 0, 100)
    chin_per = map_value(chin_val, chin_min, chin_max, 0, 100)
    left_eye_per = map_value(eyes_left_val, eyes_min, eyes_max, 0, 100)
    right_eye_per = map_value(eyes_right_val, eyes_min, eyes_max, 0, 100)

    upper = round((forehead_per + right_eye_per + left_eye_per)/ 3)
    middle = round((cheek_left_per + cheek_right_per + nose_per)/ 3)
    lower = round(chin_per)
    str_print = "Upper Region: {}%\nMiddle Region: {}%\nLower Region: {}%".format(upper, middle, lower)
    print('-' * 20, 'WRINKLES ANALYSIS', '-' * 40)
    print(str_print)
    print('-' * 77)
    return dict(wrinkles=dict(upper_part = upper,middle_part = middle, lower_part = lower))


def lips_analysis_process(reg_dict):
    from random import randint
    lips_cropped = reg_dict['lips']['roi_image']

    edge_feature = []
    r_feature = []
    g_feature = []
    b_feature = []
    moisture_label = []

    # Get Features from parameters
    with open(txt_file_moisture) as myfile:
        for line in myfile:
            f1, f2, f3, f4, l = map(float, line.split(','))
            edge_feature.append(f1)
            r_feature.append(f2)
            g_feature.append(f3)
            b_feature.append(f4)
            moisture_label.append(l)

    features = zip(edge_feature, r_feature, g_feature, b_feature)
    features = list(features)

    # Creating GaussianNB model
    model = GaussianNB()

    edge_value = get_edge_value_lips(lips_cropped)
    # edge_feature.append(edge_value)

    r_value = average_color(lips_cropped)[0]
    g_value = average_color(lips_cropped)[1]
    b_value = average_color(lips_cropped)[2]
    model.fit(features, moisture_label)

    test_list = [[edge_value, r_value, g_value, b_value]]
    predicted = model.predict(test_list)
    predicted_proba = model.predict_proba(test_list)
    if predicted[0] == 1.0:
        percent = int(100 * predicted_proba[0][1])
    else:
        percent = randint(10,30)
    print('-' * 20, 'LIPS MOISTURE ANALYSIS', '-' * 40)
    print('Lips Moisture',str(percent)+'%')
    print('-' * 77)
    return dict(moisture=dict(lips=percent))
