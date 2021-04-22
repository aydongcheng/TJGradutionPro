import glob
import cv2
import numpy as np
import math
from collections import defaultdict
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import face_recognition
import random


def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    imshow(origin_img)


def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


def resize_image(image_array, size):
    face_locations = face_recognition.face_locations(image_array)
    if len(face_locations) == 0:
        return []
    top, right, bottom, left = face_locations[0]
    face_size = (bottom - top) * (right - left)
    ratio = pow(face_size / 0.7 / (size ** 2), 0.5)
    new_size = (int(image_array.shape[1] / ratio), int(image_array.shape[0] / ratio))
    return cv2.resize(image_array, new_size)


def corp_face(image_array, size, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param size: single int value, size for w and h after crop
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    left, top: left and top coordinates of cropping
    """
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - size / 2, x_center + size / 2)

    eye_landmark = landmarks['left_eye'] + landmarks['right_eye']
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = landmarks['top_lip'] + landmarks['bottom+lip']
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top, bottom = eye_center[1] - (size - mid_part) / 2, lip_center[1] + (size - mid_part) / 2

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top


def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


imges = []
count = 0
for jpgfile in glob.glob(r'D:\demo\PyPro\TJGradutionProData\img\*.jpg'):
    img = cv2.cvtColor(cv2.imread(jpgfile), cv2.COLOR_RGB2BGR)
    if 90000 < img.shape[0] * img.shape[1]:
        img_name = jpgfile.split('\\')[-1]
        image_array = np.array(img, dtype=np.uint8)
        image_array = resize_image(image_array, 200)
        if len(image_array) == 0:
            continue
        # imshow(image_array)
        # plt.show()

        face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
        if len(face_landmarks_list) == 0:
            continue
        face_landmarks_dict = face_landmarks_list[0]
        # print(face_landmarks_dict, end=" ")
        # visualize_landmark(image_array=image_array, landmarks=face_landmarks_dict)
        # plt.show()
        aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks=face_landmarks_dict)
        # Image.fromarray(np.hstack((image_array, aligned_face)))
        # visualize_landmark(image_array=aligned_face, landmarks=face_landmarks_dict)
        # plt.show()
        rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                             eye_center=eye_center, angle=angle, row=image_array.shape[0])
        # visualize_landmark(image_array=aligned_face, landmarks=rotated_landmarks)
        # plt.show()
        cropped_face, left, top = corp_face(image_array=aligned_face, size=200, landmarks=rotated_landmarks)
        finish_img = Image.fromarray(cropped_face)
        finish_img.save(r'D:\demo\PyPro\TJGradutionProData\img-cropped\{0}.jpg'.format(img_name))
        count += 1
        # plt.imshow(Image.fromarray(cropped_face))
        # transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top)
        # visualize_landmark(image_array=cropped_face, landmarks=transferred_landmarks)
        # plt.show()
