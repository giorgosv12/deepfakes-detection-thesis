import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from Dataset_Preprocessing.create_masters import create_masters as create_master


class Vox_dataset(Dataset):
    """
    Dataset class that will be used in all the models.
    """

    def __init__(self, folders_paths, num_frames, create_masters=False, delete_masters=False, skip_checks=False,
                 train_for_all_data=False,  search_text_original = 'original',  return_label=False):
        """
        Initialize the dataset.
        :param folders_paths: List of paths to the folders containing the npz files for each video.
        :param num_frames: Number of frames to be loaded for each video.
        :param create_masters: Boolean flag for creating a master npz file for each video containing all the npz files.
        :param delete_masters: Boolean flag for deleting existing master npz files.
        :param skip_checks: Boolean flag for skipping existence checks for master npz files.
        :param train_for_all_data: Boolean flag for using all frames in the dataset for training
        :param search_text_original: text to search for at paths that original videos must contain. Default is original
        :param return_label: return labels (0 for original, 1 for fake) alongside the values. Default is False
        """
        self.folders_paths = folders_paths
        f1 = folders_paths.copy()
        self.length = len(folders_paths)
        self.num_frames = num_frames
        self.train_for_all_data = train_for_all_data
        self.scaler = MinMaxScaler()
        self.search_text_original = search_text_original
        self.return_label=return_label

        for i in range(0, len(folders_paths)):
            folders_paths[i] = folders_paths[i].split('____')[0]

        if delete_masters and not create_masters:
            raise Exception("Cannot delete masters without then creating them")

        if delete_masters:
            for folder_path in folders_paths:
                master_path = os.path.join(folder_path, 'master.npz')
                if not skip_checks:
                    if os.path.exists(master_path):
                        os.remove(master_path)
                    else:
                        print("Can not find master file in folder: ", folder_path)
                        exit()

        if not create_masters:
            for folder_path in folders_paths:
                master_path = os.path.join(folder_path, 'master.npz')
                if not skip_checks:
                    if not os.path.exists(master_path):
                        print(master_path)
                        raise Exception("create_masters=False but not all master.npz files located."
                                        " Try setting create_masters=True")

        if create_masters:
            for folder_path in folders_paths:
                master_path = os.path.join(folder_path, 'master.npz')
                if not skip_checks:
                    if os.path.exists(master_path):
                        raise Exception("create_masters=True but some master.npz files were located."
                                        " Try setting delete_masters=True")

            print('Creating master.npz files. This may take a while...')
            for path in folders_paths:
                create_master(path, remove=False)

        self.folders_paths = f1

    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        :param index: Index of the item to be retrieved.
        :return: A tensor containing the selected frames from the video.
        """
        if not self.train_for_all_data:
            master_path = os.path.join(self.folders_paths[index], 'master.npz')
            with np.load(master_path) as data:
                exp = data['exp']
                pose = data['pose']
                shape = data['shape']
                # tex = data['tex']
                # angles = data['angles']
                # mocs = data['mocs'].reshape(-1, 1)
                # left_eye = data['left_eye'].reshape(-1, 1)
                # right_eye = data['right_eye'].reshape(-1, 1)

            max_rand = exp.shape[0] - self.num_frames
            start_idx = random.randint(0, max_rand)

        else:
            splitted = self.folders_paths[index].split('____')
            folder_path = splitted[0]
            start_idx = int(splitted[1])
            master_path = os.path.join(folder_path, 'master.npz')
            with np.load(master_path) as data:
                exp = data['exp']
                pose = data['pose']
                shape = data['shape']
                # tex = data['tex']
                # angles = data['angles']
                # mocs = data['mocs'].reshape(-1, 1)
                # left_eye = data['left_eye'].reshape(-1, 1)
                # right_eye = data['right_eye'].reshape(-1, 1)

        mat = np.concatenate((exp[start_idx:start_idx + self.num_frames], pose[start_idx:start_idx + self.num_frames],
                              shape[start_idx:start_idx + self.num_frames]), axis=1)

        #   # The following segment returns all the features that are calculated
        # mat = np.concatenate((exp[start_idx:start_idx + self.num_frames],
        #                       pose[start_idx:start_idx + self.num_frames],
        #                       shape[start_idx:start_idx + self.num_frames],
        #                       angles[start_idx:start_idx + self.num_frames],
        #                       mocs[start_idx:start_idx + self.num_frames],
        #                       left_eye[start_idx:start_idx + self.num_frames],
        #                       right_eye[start_idx:start_idx + self.num_frames]),
        #                      axis=1)

        mat = self.scaler.fit_transform(mat)
        mat = np.array([mat])

        if not self.return_label:
            return torch.from_numpy(mat)
        else:
            if master_path.find(self.search_text_original) != -1:  # 0 is original video
                return torch.from_numpy(mat), 0
            else:
                return torch.from_numpy(mat), 1

    def __len__(self):
        """
        Get length of the dataset
        :returns: dataset's length
        """
        return self.length
