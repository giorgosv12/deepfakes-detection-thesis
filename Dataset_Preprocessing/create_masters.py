import os
from fnmatch import fnmatch
import numpy as np


def get_angle(index, point_a, point_b):
    """
    Calculates the angle between two vectors, both of which originate from a common point (index). The two vectors are
    defined by point_a and point_b, and their respective positions relative to the index point.

    :param index: The common point from which the two vectors originate
    :param point_a: The position of the first point defining the first vector
    :param point_b: The position of the second point defining the second vector
    :returns: The angle between the two vectors in radians
    """
    vector_1 = point_a - index
    vector_2 = point_b - index
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)

def ret_feat(landmarks):
    """
    Calculates features used for detecting deepfakes (described in chapter 3.1 of thesis) from the 68 facial landmarks.

    :param landmarks: 68 facial landmarks positions
    :return tuple containing features described in chapter 3.1 of thesis
    """
    angles = []
    idx = 3

    landmarks = landmarks.squeeze()
    # calculate mouth open/closed score
    a = landmarks[51,:idx]
    b = landmarks[57,:idx]
    c = landmarks[62,:idx]
    d = landmarks[66,:idx]
    ar = np.linalg.norm(c - d)
    par = np.linalg.norm(a - b)
    mocs = ar / par

    # calculate eye aspect ratio for theleft eye
    l1 = landmarks[36,:idx]
    l2 = landmarks[37,:idx]
    l3 = landmarks[38,:idx]
    l4 = landmarks[39,:idx]
    l5 = landmarks[40, :idx]
    l6 = landmarks[41, :idx]
    ear_l = (np.linalg.norm(l2 - l6) + np.linalg.norm(l3 - l5))/(2 * np.linalg.norm(l1 - l4))

    # calculate eye aspect ratio for the right eye
    r1 = landmarks[42,:idx]
    r2 = landmarks[43,:idx]
    r3 = landmarks[44,:idx]
    r4 = landmarks[45,:idx]
    r5 = landmarks[46, :idx]
    r6 = landmarks[47, :idx]
    ear_r = (np.linalg.norm(r2 - r6) + np.linalg.norm(r3 - r5))/(2 * np.linalg.norm(r1 - r4))

    # calculate angles described in diagram 1 from image 17 of thesis
    angles.append(get_angle(landmarks[1], landmarks[27], landmarks[28]))
    angles.append(get_angle(landmarks[2], landmarks[28], landmarks[29]))
    angles.append(get_angle(landmarks[3], landmarks[29], landmarks[30]))
    angles.append(get_angle(landmarks[4], landmarks[30], landmarks[33]))
    angles.append(get_angle(landmarks[12], landmarks[33], landmarks[30]))
    angles.append(get_angle(landmarks[13], landmarks[30], landmarks[29]))
    angles.append(get_angle(landmarks[14], landmarks[29], landmarks[28]))
    angles.append(get_angle(landmarks[15], landmarks[28], landmarks[27]))

    #calculate angles described in diagram 2 from image 17 of thesis
    angles.append(get_angle(landmarks[1], landmarks[36], landmarks[41]))
    angles.append(get_angle(landmarks[2], landmarks[41], landmarks[40]))
    angles.append(get_angle(landmarks[3], landmarks[40], landmarks[39]))
    angles.append(get_angle(landmarks[13], landmarks[42], landmarks[47]))
    angles.append(get_angle(landmarks[14], landmarks[47], landmarks[46]))
    angles.append(get_angle(landmarks[15], landmarks[46], landmarks[45]))

    #calculate angles described in diagram 3 from image 17 of thesis
    angles.append(get_angle(landmarks[3], landmarks[31], landmarks[32]))
    angles.append(get_angle(landmarks[4], landmarks[32], landmarks[33]))
    angles.append(get_angle(landmarks[12], landmarks[33], landmarks[34]))
    angles.append(get_angle(landmarks[13], landmarks[34], landmarks[35]))

    return mocs, ear_l, ear_r, np.array(angles)


def create_masters(folder_path, remove):
    """
    This function takes in a folder path containing NPZ files and creates a master NPZ file containing all the data from
    the individual NPZ files. It also provides an option to remove the individual NPZ files after creating the master
    NPZ file.

    :param folder_path: path to the folder containing the individual NPZ files
    :param remove:  flag to specify whether to remove the individual NPZ files after creating the master NPZ file
    :returns: None
    """
    npz_paths = []

    # Initializing empty lists for all the data that will be extracted from the individual NPZ files
    exp = []
    pose = []
    shape = []
    tex = []
    mocs = []
    left_eye = []
    right_eye = []
    angles = []

    # Iterating through the folder and its subdirectories to find all NPZ files
    for path2, subdirs, files in os.walk(folder_path):
        for name in files:
            if fnmatch(name, '*.npz'):
                npz_paths.append(os.path.join(path2, name))
    npz_paths.sort()

    # Iterating through the found NPZ files and extracting the data from them
    for npz_path in npz_paths:
        with np.load(npz_path) as data:
            exp.append(data['exp'][0])
            pose.append(data['pose'][0])
            shape.append(data['shape'][0])
            tex.append(data['tex'][0])
            landmarks = data['landmarks']
            mocs1, left1, right1, angle1 = ret_feat(landmarks)
            mocs.append(mocs1)
            left_eye.append(left1)
            right_eye.append(right1)
            angles.append(angle1)

    # Saving all the extracted data into a single NPZ file named 'master.npz' in the given folder
    np.savez(os.path.join(folder_path, 'master.npz'), exp=np.array(exp), pose=np.array(pose), shape=np.array(shape),
             tex=np.array(tex),
             mocs=np.array(mocs), left_eye=np.array(right_eye),
             right_eye=np.array(left_eye), angles=np.array(angles))

    # Checking if the user wants to remove the individual NPZ files
    if remove:
        for npz_path in npz_paths:
            os.remove(npz_path)
