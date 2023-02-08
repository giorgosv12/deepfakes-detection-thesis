import torch
import numpy as np
from Utils.dataset import Vox_dataset

def load_data(opt):
    """
    Loads training and testing data

    :param opt: argument parser object containing the configuration for the model
    :type opt: argparse.Namespace
    :return: Dictionary of training and testing data loaders
    :rtype: dict[str, torch.utils.data.DataLoader]
    """

    data_folders = np.load('./data_folders_paths.npy')

    train_paths = data_folders
    test_paths = np.load('./ff_only_orig_deepfakes.npy')

    splits = ['train', 'test']
    shuffle = {'train': True, 'test': True}
    dataset = {}
    dataset['train'] = Vox_dataset(folders_paths=train_paths, num_frames=96, skip_checks=True,
                                   return_label=True, search_text_original='voxceleb')
    dataset['test'] =Vox_dataset(folders_paths=test_paths, num_frames=96, skip_checks=True,
                                   return_label=True, search_text_original='original', )


    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.train_batch,
                                                 shuffle=shuffle[x],
                                                 num_workers=opt.workers,
                                                 drop_last = True) for x in splits}
    return dataloader
