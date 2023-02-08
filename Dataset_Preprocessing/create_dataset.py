import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
from Flame_2020 import FLAME, get_config
from create_masters import create_masters
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pathlib

from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

def main(root, savepath, replace_photos_with_cropped=False, iscrop=True, device='cuda:0', useTex=False,
         rasterizer_type='pytorch3d'):
    """
    Process for extracting FLAME model coefficients and other features used for detecting deepfakes (described in
    chapter 3.1 of thesis) from videos. These features are saved in a master.npz file for each video.

    :param root: filepath of file containing the paths of videos to process
    :param savepath: path of folder where files will be saved
    :param replace_photos_with_cropped: whether to replaced extracted video frames with cropped versions. Default is
        False
    :param iscrop: whether video is at correct resolution (224x224). Default is True
    :param device: device to run models and store data. Default is cuda:0
    :param useTex: whether to use texture for deca model. Default is False
    :param rasterizer_type: type of rasterizer that will be used. Default is pytorch3d
    """

    # set the parameters for deca model and initialize the deca model to specified device
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    # initialize flame model
    flamelayer = FLAME(get_config())

    # load paths of videos to process
    paths = np.load(root)

    for path in paths:
        print("Current Video: ", path)

        # process each video frame and extract deca coefficients and facial landmarks from it
        deca_video(path, savepath, deca, flamelayer, replace_photos_with_cropped, iscrop,  device)

        # calculate extra parameters used in thesis for deepfakes detection and save them alongside flame model and
        # texture coefficients to a new master.npz file
        pathh = pathlib.PurePath(os.path.splitext(path)[0])
        create_masters(os.path.join(savepath, pathh.parent.name, pathh.name), remove=True)

        # delete path of processed video file from list of paths
        k = np.load(root)
        new_arr = np.delete(k, np.where(k == path))
        np.save(root, new_arr)

def deca_video(inputpath, savepath, deca, flamelayer, replace_photos_with_cropped=False, iscrop=False, device='cuda:0'):
    """
        Process a single video by extracting facial landmarks and other information using the Dataset_Preprocessing and FLAME models.

        :param inputpath: input video path
        :param savepath:  path to the folder where the processed data will be saved
        :param deca: deca model to use
        :param flamelayer: flame model to use
        :param replace_photos_with_cropped: whether to replaced extracted video frames with cropped versions. Default is
        False
        :param iscrop: Whether to crop the frames before processing, by default False
        :param device: The device to use for processing (e.g. 'cuda:0'), by default 'cuda:0'
        """

    # Create a TestData object from the input video and load it into a data loader
    testdata = datasets.TestData(inputpath, iscrop=iscrop)
    dataloader = DataLoader(testdata, batch_size=1, pin_memory=True)

    # Create a save folder for the processed data
    pathh = pathlib.PurePath(os.path.splitext(inputpath)[0])
    savefolder = os.path.join(savepath, pathh.parent.name, pathh.name)
    os.makedirs(savefolder, exist_ok=True)

    # Iterate through the video frames
    for i, data in enumerate(tqdm(dataloader)):

        # Extract the image name and image data
        name = data['imagename'][0]
        images = data['image'].to(device)[None, ...][0]

        # Optionally replace the original frame with a cropped version
        if replace_photos_with_cropped:
            os.remove(os.path.join(savefolder, name) + '.jpg')
            images.squeeze().cpu().numpy()
            plt.imsave(os.path.join(savefolder, name) + '.jpg')

        with torch.no_grad():

            # Encode the image data using the Dataset_Preprocessing model
            codedict = deca.encode(images)

            # Run flame model for the parameters extracted from the image and get the 3D facial landmarks
            _, landmarks = flamelayer(shape_params=codedict['shape'].cpu(), expression_params=codedict['exp'].cpu(),
                                      pose_params=codedict['pose'].cpu())

            # move Dataset_Preprocessing coefficients and facial landmarks to the cpu and transform the torch tensors to numpy arrays to
            # be saved
            shape_coeffs = codedict['shape'].cpu().numpy()
            pose_coeffs = codedict['pose'].cpu().numpy()
            exp_coeffs = codedict['exp'].cpu().numpy()
            tex = codedict['tex'].cpu().numpy()
            landmarks_3d = landmarks.cpu().numpy()

        # Save coefficients and facial landmarks to a npz file
        np.savez(os.path.join(savefolder, name) + '.npz', shape=shape_coeffs, pose=pose_coeffs, exp=exp_coeffs, tex=tex,
                 landmarks=landmarks_3d)

if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='List of video files paths')
    parser.add_argument('-dev', type=str, required=True,
                        help='device to use for processing (e.g. cuda:0)')
    parser.add_argument('-s', type=str, required=False, default='./output',
                        help='path to the folder where the processed data will be saved')
    args = parser.parse_args()

    # execute main method for input arguments
    main(root=args.i, device=args.dev, savepath=args.s)
