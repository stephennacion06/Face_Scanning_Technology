import cv2
import dlib
import time

# path to change
absolutePath = '/home/stephen/THESIS_PROTOTYPE_DEV/FACE_SCANNING_TECHNOLOGY_master/'

# file paths
predictor_path = absolutePath + 'shape_predictor_68_face_landmarks.dat'
face_cascade = cv2.CascadeClassifier(absolutePath + 'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(absolutePath + 'haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# file paths to be saved
lc_roi_path = absolutePath + 'Client_Regions/left_cheek.jpg'
rc_roi_path = absolutePath + 'Client_Regions/right_cheek.jpg'
nose_roi_path = absolutePath + 'Client_Regions/nose.jpg'
forehead_roi_path = absolutePath + 'Client_Regions/forehead.jpg'
chin_roi_path = absolutePath + 'Client_Regions/chin.jpg'
left_eye_roi_path = absolutePath + 'Client_Regions/left_eye.jpg'
right_eye_roi_path = absolutePath + 'Client_Regions/right_eye.jpg'
lips_roi_path = absolutePath + 'Client_Regions/lips.jpg'
face_roi_path = absolutePath + 'Client_Regions/face.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def save_forehead(img_input):
    img = img_input
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eye_index = 0
    adjusted_y = 10
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            if eye_index == 0:
                l_eye_x = ex
                l_eye_y = ey
                l_eye_w = ew
                l_eye_h = eh

            else:
                r_eye_x = ex
                r_eye_y = ey
                r_eye_w = ew
                r_eye_h = eh
            eye_index += 1

        if l_eye_x > r_eye_x:
            l_eye_x_temp = l_eye_x
            l_eye_y_temp = l_eye_y
            l_eye_w_temp = l_eye_w
            l_eye_h_temp = l_eye_h

            l_eye_x = r_eye_x
            l_eye_y = r_eye_y
            l_eye_w = r_eye_w
            l_eye_h = r_eye_h

            r_eye_x = l_eye_x_temp
            r_eye_y = l_eye_y_temp
            r_eye_w = l_eye_w_temp
            r_eye_h = l_eye_h_temp

        forehead = roi_color[adjusted_y:l_eye_y - adjusted_y, l_eye_x:(r_eye_x + l_eye_x)]
        forehead_dict = {'roi_image': forehead, 'org_x': l_eye_x, 'org_y': adjusted_y, 'width': r_eye_x + l_eye_x,
                         'height': l_eye_y - adjusted_y, 'roi_face': roi_color}

    return forehead_dict


def save_eye_left_right(input_img):
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

            if n == 17:
                l_eye_origin_y = y
                r_eye_origin_y = y
            elif n == 1:
                l_eye_height = r_eye_height = y
            elif n == 0:
                l_eye_origin_x = x

            elif n == 36:
                l_eye_width = x

            elif n == 45:
                r_eye_origin_x = x

            elif n == 16:
                r_eye_width = x

    roi_eye_l = frame[l_eye_origin_y:l_eye_height, l_eye_origin_x:l_eye_width]
    roi_eye_r = frame[r_eye_origin_y:r_eye_height, r_eye_origin_x:r_eye_width]
    l_eye_dict = {'roi_image': roi_eye_l, 'org_x': l_eye_origin_x, 'org_y': l_eye_origin_y, 'width': l_eye_width,
                  'height': l_eye_height}
    r_eye_dict = {'roi_image': roi_eye_r, 'org_x': r_eye_origin_x, 'org_y': r_eye_origin_y, 'width': r_eye_width,
                  'height': r_eye_height}
    eyes_dict = {'eye_left': l_eye_dict, 'eye_right': r_eye_dict}

    return eyes_dict


def save_cheeks_nose_chin(input_img):
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

            # CHEEKS
            if n == 28:
                left_cheek_origin_y = right_cheek_origin_y = y
            elif n == 17:
                left_cheek_origin_x = x
            elif n == 4:
                left_cheek_height = right_cheek_height = y
            elif n == 49:
                left_cheek_width = x
            elif n == 42:
                right_cheek_origin_x = x
            elif n == 26:
                right_cheek_width = x

            # NOSE
            elif n == 31:
                nose_origin_x = x
            elif n == 27:
                nose_origin_y = y
            elif n == 35:
                nose_width = x
            elif n == 33:
                nose_height = y

            # CHIN
            elif n == 57:
                chin_origin_y = y
            elif n == 5:
                chin_origin_x = x
            elif n == 11:
                chin_width = x
            elif n == 8:
                chin_height = y

    left_cheek_cropped = frame[left_cheek_origin_y:left_cheek_height, left_cheek_origin_x:left_cheek_width]
    right_cheek_cropped = frame[right_cheek_origin_y:right_cheek_height, right_cheek_origin_x:right_cheek_width]

    nose_cropped = frame[nose_origin_y:nose_height, nose_origin_x:nose_width]

    chin_cropped = frame[chin_origin_y:chin_height, chin_origin_x:chin_width]

    l_cheek_dict = {'roi_image': left_cheek_cropped, 'org_x': left_cheek_origin_x, 'org_y': left_cheek_origin_y,
                    'width': left_cheek_width,
                    'height': left_cheek_height}
    r_cheek_dict = {'roi_image': right_cheek_cropped, 'org_x': right_cheek_origin_x, 'org_y': right_cheek_origin_y,
                    'width': right_cheek_width,
                    'height': right_cheek_height}
    nose_dict = {'roi_image': nose_cropped, 'org_x': nose_origin_x, 'org_y': nose_origin_y, 'width': nose_width,
                 'height': nose_height}
    chin_dict = {'roi_image': chin_cropped, 'org_x': chin_origin_x, 'org_y': chin_origin_y, 'width': chin_width,
                 'height': chin_height}

    front_face_dict = dict(cheek_left=l_cheek_dict, cheek_right=r_cheek_dict, nose=nose_dict, chin=chin_dict)

    return front_face_dict


def find_lips(input_img):
    input_img = cv2.resize(input_img, (600, 600), interpolation=cv2.INTER_AREA)
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
    chin_dict = {'roi_image': lips_cropped, 'org_x': lips_origin_x, 'org_y': lips_origin_y, 'width': lips_width,
                 'height': lips_height}

    return chin_dict


def find_8_regions(input_img, time_now):
    # return the matrix of cropped forehead, side eyes, nose, cheeks, chin and lips

    forehead_cropped = save_forehead(input_img)
    roi_eye_cropped_l, roi_eye_cropped_r = save_eye_left_right(input_img)
    nose_cropped, left_cheek_cropped, right_cheek_cropped, chin_cropped = save_cheeks_nose_chin(input_img)
    lips_cropped = find_lips(input_img)

    return forehead_cropped, nose_cropped, left_cheek_cropped, right_cheek_cropped, chin_cropped, roi_eye_cropped_l, roi_eye_cropped_r, lips_cropped


def display_regions(frame, regions_dict):
    bold = 2
    cv2.rectangle(regions_dict['forehead']['roi_face'],
                  (regions_dict['forehead']['org_x'], regions_dict['forehead']['org_y']),
                  (regions_dict['forehead']['width'], regions_dict['forehead']['height']),
                  (0, 255, 0), bold)
    cv2.rectangle(frame,
                  (regions_dict['lips']['org_x'], regions_dict['lips']['org_y']),
                  (regions_dict['lips']['width'], regions_dict['lips']['height']),
                  (0, 255, 0), bold)
    cv2.rectangle(frame,
                  (
                      regions_dict['front_face']['cheek_left']['org_x'],
                      regions_dict['front_face']['cheek_left']['org_y']),
                  (regions_dict['front_face']['cheek_left']['width'],
                   regions_dict['front_face']['cheek_left']['height']),
                  (0, 255, 0), bold)
    cv2.rectangle(frame,
                  (regions_dict['front_face']['cheek_right']['org_x'],
                   regions_dict['front_face']['cheek_right']['org_y']),
                  (regions_dict['front_face']['cheek_right']['width'],
                   regions_dict['front_face']['cheek_right']['height']),
                  (0, 255, 0), bold)
    cv2.rectangle(frame,
                  (regions_dict['front_face']['nose']['org_x'], regions_dict['front_face']['nose']['org_y']),
                  (regions_dict['front_face']['nose']['width'], regions_dict['front_face']['nose']['height']),
                  (0, 255, 0), bold)
    cv2.rectangle(frame,
                  (regions_dict['front_face']['chin']['org_x'], regions_dict['front_face']['chin']['org_y']),
                  (regions_dict['front_face']['chin']['width'], regions_dict['front_face']['chin']['height']),
                  (0, 255, 0), bold)
    cv2.rectangle(frame,
                  (regions_dict['side_eyes']['eye_left']['org_x'], regions_dict['side_eyes']['eye_left']['org_y']),
                  (regions_dict['side_eyes']['eye_left']['width'], regions_dict['side_eyes']['eye_left']['height']),
                  (0, 255, 0), bold)
    cv2.rectangle(frame,
                  (regions_dict['side_eyes']['eye_right']['org_x'], regions_dict['side_eyes']['eye_right']['org_y']),
                  (regions_dict['side_eyes']['eye_right']['width'], regions_dict['side_eyes']['eye_right']['height']),
                  (0, 255, 0), bold)

    cv2.imshow('Pores Analysis', frame)
    cv2.waitKey(0)


def regions3_display(frame, regions_dict):
    bold = 2
    cv2.rectangle(frame,
                  (regions_dict['side_eyes']['eye_left']['org_x'], regions_dict['forehead']['height']),
                  (regions_dict['side_eyes']['eye_right']['width'], regions_dict['front_face']['cheek_left']['org_y']),
                  (0, 255, 0), bold)

    cv2.rectangle(frame,
                  (regions_dict['side_eyes']['eye_left']['org_x'], regions_dict['front_face']['cheek_left']['org_y']),
                  (regions_dict['side_eyes']['eye_right']['width'], regions_dict['lips']['org_y']),
                  (0, 255, 0), bold)

    cv2.rectangle(frame,
                  (regions_dict['front_face']['cheek_left']['org_x'], regions_dict['lips']['org_y']),
                  (regions_dict['front_face']['cheek_right']['width'], regions_dict['front_face']['chin']['height']),
                  (0, 255, 0), bold)

    cv2.imshow('Face Scanning Technology', frame)
    cv2.waitKey(0)


def regions3_display_label(frame, regions_dict, p_dict, w_dict, l_dict):
    cv2.imshow('Face Scanning Technology',frame)
    cv2.waitKey(0)

    bold = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .7
    color = (255, 0, 0)
    thickness = 2
    adjust_y = 40

    color_l = (255, 0, 0)
    thickness_l = 2

    upper_p_str_ = 'Pores: ' + str(p_dict['pores']['upper_part']) + '%'
    middle_p_str_ = 'Pores: ' + str(p_dict['pores']['middle_part']) + '%'
    lower_p_str_ = 'Pores: ' + str(p_dict['pores']['lower_part']) + '%'

    upper_w_str_ = 'Wrinkles: ' + str(w_dict['wrinkles']['upper_part']) + '%'
    middle_w_str_ = 'Wrinkles: ' + str(w_dict['wrinkles']['middle_part']) + '%'
    lower_w_str_ = 'Wrinkles: ' + str(w_dict['wrinkles']['lower_part']) + '%'

    moisture_str_ = 'Lips Moisture: ' + str(l_dict['moisture']['lips']) + '%'

    cv2.rectangle(frame,
                  (regions_dict['side_eyes']['eye_left']['org_x'], regions_dict['forehead']['height']),
                  (regions_dict['side_eyes']['eye_right']['width'], regions_dict['front_face']['cheek_left']['org_y']),
                  (0, 255, 0), bold)
    cv2.putText(frame, upper_p_str_,
                (regions_dict['side_eyes']['eye_right']['width'] + 5,
                 regions_dict['forehead']['height']),
                font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, upper_w_str_,
                (regions_dict['side_eyes']['eye_right']['width'] + 5,
                 regions_dict['forehead']['height']+adjust_y ) ,
                font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.rectangle(frame,
                  (regions_dict['side_eyes']['eye_left']['org_x'], regions_dict['front_face']['cheek_left']['org_y']),
                  (regions_dict['side_eyes']['eye_right']['width'], regions_dict['lips']['org_y']),
                  (0, 255, 0), bold)
    cv2.putText(frame, middle_p_str_,
                (regions_dict['side_eyes']['eye_right']['width'] + 5,
                 regions_dict['front_face']['cheek_left']['org_y']),
                font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, middle_w_str_,
                (regions_dict['side_eyes']['eye_right']['width'] + 5,
                 regions_dict['front_face']['cheek_left']['org_y'] + adjust_y ),
                font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.rectangle(frame,
                  (regions_dict['front_face']['cheek_left']['org_x'], regions_dict['lips']['org_y']),
                  (regions_dict['front_face']['cheek_right']['width'], regions_dict['front_face']['chin']['height']),
                  (0, 255, 0), bold)

    cv2.putText(frame, lower_p_str_,
                (regions_dict['front_face']['cheek_right']['width'] + 5,
                 regions_dict['lips']['org_y'] + adjust_y),
                font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, lower_w_str_,
                (regions_dict['front_face']['cheek_right']['width'] + 5,
                 regions_dict['lips']['org_y'] + adjust_y + adjust_y ),
                font, fontScale, color, thickness, cv2.LINE_AA)

    # LINE FOR MOISTURE
    start_point = (int((regions_dict['lips']['org_x']+regions_dict['lips']['width'])/2)
                   , int((regions_dict['lips']['org_y']+regions_dict['lips']['height'])/2))
    end_point = (regions_dict['front_face']['chin']['org_x'],
                 regions_dict['front_face']['chin']['height']+20)

    cv2.line(frame, start_point, end_point, color_l, thickness_l)

    cv2.putText(frame, moisture_str_,
                (regions_dict['front_face']['chin']['org_x']-20,
                 regions_dict['front_face']['chin']['height']+50),
                font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Face Scanning Technology', frame)
    cv2.waitKey(0)


def save_regions(reg_dict):
    cv2.imwrite(lc_roi_path, reg_dict['front_face']['cheek_left']['roi_image'])
    cv2.imwrite(rc_roi_path, reg_dict['front_face']['cheek_right']['roi_image'])
    cv2.imwrite(nose_roi_path, reg_dict['front_face']['nose']['roi_image'])
    cv2.imwrite(chin_roi_path, reg_dict['front_face']['chin']['roi_image'])
    cv2.imwrite(forehead_roi_path, reg_dict['forehead']['roi_image'])
    cv2.imwrite(left_eye_roi_path, reg_dict['side_eyes']['eye_left']['roi_image'])
    cv2.imwrite(right_eye_roi_path, reg_dict['side_eyes']['eye_right']['roi_image'])
    cv2.imwrite(lips_roi_path, reg_dict['lips']['roi_image'])
    cv2.imwrite(face_roi_path, reg_dict['forehead']['roi_face'])


def save_time_regions(reg_dict, time_now):
    lc_time_path_p = absolutePath + 'training_data/pores/cheek_pictures/left_cheek_{}.jpg'.format(time_now)
    rc_time_path_p = absolutePath + 'training_data/pores/cheek_pictures/right_cheek_{}.jpg'.format(time_now)
    nose_time_path_p = absolutePath + 'training_data/pores/nose_pictures/nose_{}.jpg'.format(time_now)
    forehead_time_path_p = absolutePath + 'training_data/pores/forehead_pictures/forehead_{}.jpg'.format(time_now)
    chin_time_path_p = absolutePath + 'training_data/pores/chin_pictures/chin_{}.jpg'.format(time_now)
    left_eye_time_path_p = absolutePath + 'training_data/pores/side_eyes/left_eye_{}.jpg'.format(time_now)
    right_eye_time_path_p = absolutePath + 'training_data/pores/side_eyes/right_eye_{}.jpg'.format(time_now)

    lc_time_path_w = absolutePath + 'training_data/wrinkles/cheek_pictures/left_cheek_{}.jpg'.format(time_now)
    rc_time_path_w = absolutePath + 'training_data/wrinkles/cheek_pictures/right_cheek_{}.jpg'.format(time_now)
    nose_time_path_w = absolutePath + 'training_data/wrinkles/nose_pictures/nose_{}.jpg'.format(time_now)
    forehead_time_path_w = absolutePath + 'training_data/wrinkles/forehead_pictures/forehead_{}.jpg'.format(time_now)
    chin_time_path_w = absolutePath + 'training_data/wrinkles/chin_pictures/chin_{}.jpg'.format(time_now)
    left_eye_time_path_w = absolutePath + 'training_data/wrinkles/side_eyes/left_eye_{}.jpg'.format(time_now)
    right_eye_time_path_w = absolutePath + 'training_data/wrinkles/side_eyes/right_eye_{}.jpg'.format(time_now)

    lips_time_path = absolutePath + 'training_data/moisture/cropped_lips/lips_{}.jpg'.format(time_now)

    cv2.imwrite(lc_time_path_p, reg_dict['front_face']['cheek_left']['roi_image'])
    cv2.imwrite(rc_time_path_p, reg_dict['front_face']['cheek_right']['roi_image'])
    cv2.imwrite(nose_time_path_p, reg_dict['front_face']['nose']['roi_image'])
    cv2.imwrite(chin_time_path_p, reg_dict['front_face']['chin']['roi_image'])
    cv2.imwrite(forehead_time_path_p, reg_dict['forehead']['roi_image'])
    cv2.imwrite(left_eye_time_path_p, reg_dict['side_eyes']['eye_left']['roi_image'])
    cv2.imwrite(right_eye_time_path_p, reg_dict['side_eyes']['eye_right']['roi_image'])

    cv2.imwrite(lc_time_path_w, reg_dict['front_face']['cheek_left']['roi_image'])
    cv2.imwrite(rc_time_path_w, reg_dict['front_face']['cheek_right']['roi_image'])
    cv2.imwrite(nose_time_path_w, reg_dict['front_face']['nose']['roi_image'])
    cv2.imwrite(chin_time_path_w, reg_dict['front_face']['chin']['roi_image'])
    cv2.imwrite(forehead_time_path_w, reg_dict['forehead']['roi_image'])
    cv2.imwrite(left_eye_time_path_w, reg_dict['side_eyes']['eye_left']['roi_image'])
    cv2.imwrite(right_eye_time_path_w, reg_dict['side_eyes']['eye_right']['roi_image'])

    cv2.imwrite(lips_time_path, reg_dict['lips']['roi_image'])


# save_eye_left_right(frame)
if __name__ == "__main__":
    test_image_path = absolutePath + 'Pores_Analysis/face_pictures/stephen.jpg'
    frame = cv2.imread(test_image_path)
    input_img = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_AREA)
    regions_dict = {'forehead': save_forehead(input_img), 'side_eyes': save_eye_left_right(input_img),
                    'front_face': save_cheeks_nose_chin(input_img), 'lips': find_lips(input_img)}

    time_now = time.strftime("%Y%m%d-%H%M%S")
    save_regions(regions_dict)
    save_time_regions(regions_dict, time_now)
    display_regions(input_img, regions_dict)
