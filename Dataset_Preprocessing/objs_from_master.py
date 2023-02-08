import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from FlAMETex import FLAMETex
from Flame_2020 import get_config, FLAME
from renderer import Renderer


def create_obj(shape, pose, expression, tex, file_name, save_folder, flamelayer, flametex, use_tex=True):
    """
    This function creates an .obj file using the provided shape, pose, expression and texture parameters of the FLAME model.
    It uses FLAME model to generate vertex and landmark data, and a renderer to save the obj file.

    :param shape: The shape parameters of the obj file to be created
    :param pose: The pose parameters of the obj file to be created
    :param expression: The expression parameters of the obj file to be created
    :param tex: The texture parameters of the obj file to be created
    :param file_name: The name of the obj file to be saved
    :param save_folder: The folder path to save the obj file in
    :param flamelayer: A FLAME model to generate vertex and landmark data
    :param flametex: A FLAME model to generate texture data
    :param use_tex: A flag to specify whether to use texture data in the generated obj file default: True
    :returns: None
    """

    shape_params = torch.tensor(shape, dtype=torch.float32)
    pose_params = torch.tensor(pose, dtype=torch.float32)
    expression_params = torch.tensor(expression, dtype=torch.float32)

    with torch.no_grad():

        # Using FLAME model to generate vertex and landmark data
        vertice, landmark = flamelayer(shape_params, expression_params, pose_params)
        if use_tex:

            # Using FLAME model to generate texture data
            albedos = flametex(tex)

    mesh_file = './data/head_template.obj'
    render = Renderer(256,  obj_filename=mesh_file).to('cpu')
    if use_tex:

        # Saving obj file with vertex, landmark and texture data
        render.save_obj(filename=os.path.join(save_folder, file_name),
                                 vertices=vertice.squeeze().cpu(),
                                 textures=albedos.squeeze())
    else:

        # Saving obj file with only vertex and landmark data
        render.save_obj(filename=os.path.join(save_folder, file_name),
                        vertices=vertice.squeeze().cpu())



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='path of input master.npz file')
    args = parser.parse_args()

    # path of folder containing input data
    root = args.i

    # path of folder to save object created by FLAME parameters
    folder_name = 'Obj_seq'
    savefolder = os.path.splitext(root)[0]
    os.makedirs(os.path.join(savefolder, folder_name), exist_ok=True)

    # Get the configuration for the FLAME model
    config = get_config()

    # Initialize the FLAME model for vertex and landmark data
    flamelayer = FLAME(config)

    # Initialize the FLAME model for texture data
    flametex = FLAMETex('BFM', tex_path=os.path.join('./FLAME_albedo_from_BFM.npz'), n_tex=50)

    # Load the master.npz file containing all the shape, pose, expression and texture data
    with np.load(root) as data:
        tt = data['pose'].shape[0]

        # Iterate through all the data points
        for i in tqdm(range(0,tt)):
            shape = [data['shape'][i]]
            pose = [data['pose'][i]]
            expression = data['exp'][i]
            expression =[expression]
            tex = np.array([data['tex'][i]])

            # Creating obj files for each data point
            create_obj(shape,pose,expression,tex, str(i)+'.obj', os.path.join(savefolder, folder_name),
                       flamelayer, flametex, use_tex=True)

