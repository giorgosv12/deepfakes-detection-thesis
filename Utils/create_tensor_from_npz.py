import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler



def create_input(folder_path, n_frames, attrs= ['exp', 'pose', 'shape', 'tex'], start_idx=-1):
    """
    This function loads the master.npz file from the given folder path and creates an input from the attributes passed
    in the function.
        :param folder_path: The path of the folder where the master.npz file is located.
        :param n_frames: Number of frames to be included in the input.
        :param attrs: List of attributes to be included in the input. Default is ['exp', 'pose', 'shape', 'tex'].
        :param start_idx: The starting index of the frames. Default is -1, which means a random index will be selected.
        :return: Returns a torch tensor containing the input data.
    """
    master_path = os.path.join(folder_path, 'master.npz')

    with np.load(master_path) as data:
        a = []
        for i in range(0, len(attrs)):
            k = data[attrs[i]]
            if k.ndim == 1:
                k = np.reshape(k, (-1, 1))
            a.append(k)

    max_rand = a[0].shape[0] - n_frames
    if start_idx == -1:
        start_idx = random.randint(0, max_rand)

    mat = a[0][start_idx:start_idx + n_frames]
    for i in range(1,  len(a)):
        mat = np.concatenate((mat, a[i][start_idx:start_idx + n_frames]), axis=1)

    scaler = MinMaxScaler()
    a = mat
    a = scaler.fit_transform(a)
    a = np.array([[a]])

    return torch.from_numpy(a)
