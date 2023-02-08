# ----------------------------------------------------------------------------------
# This file contains all the necessary functions to test the trained One class classifiers (DenseVAE, VQVAE, OCGAN,
# GAN) for the FaceForensics++ Dataset
# The main function contains 3 modes.
#         - Mode 0: test the pretrained model for the dataset and save results to ./ForFF/ folder
#         - Mode 1: find the RMSE threshold to differentiate real from fake reconstructed data
#         - Mode 2: using the given threshold, calculate metrics for the 3 different comparison types described in chapter
#           4.3 of thesis
# ----------------------------------------------------------------------------------

import os
from fnmatch import fnmatch
from tqdm import tqdm
import numpy as np
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, \
    RocCurveDisplay, ConfusionMatrixDisplay

from Utils.create_tensor_from_npz import create_input
from Dataset_Preprocessing.create_masters import create_masters
from VAEs.models.DenseVAE import Dense_VAE
from VAEs.models.VQVAE import VQVae
from Utils.find_threshold import find_threshold
from ocgan.model.networks import get_encoder, get_decoder


def test_trained_model(saved_model_path, face_forensics_path, model_type):
    """
    Test a pre-trained model on the FaceForensics dataset and save the results to output folder.
        :param saved_model_path: path to the saved model
        :param face_forensics_path: path to the FaceForensics dataset
        :param model_type: type of model that will be tested. Can be densevae, vqvae, ocgan, gan
    """

    # inputs validation
    if model_type not in ['densevae', 'vqvae', 'ocgan', 'gan']:
        print('Wrong choice given for model type')
        exit(100)

    data_folders = list()

    # get all the folders that contain master.npz files
    for path, _, files in os.walk(face_forensics_path):
        for name in files:
            if fnmatch(name, "*master.npz"):
                if (path.find('original') != -1) or (path.find('deepfakes') != -1):
                    data_folders.append(path)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(42)

    # import the appropriate model
    if model_type == 'densevae':
        model = Dense_VAE(latent_dims=2000).to(device)
        state_dict = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(state_dict)

    elif model_type == 'vqvae':
        model = VQVae(
            num_hiddens=128, num_residual_layers=6, num_residual_hiddens=32,
            num_embeddings=512, embedding_dim=64,
            commitment_cost=0.25
                ).to(device)
        state_dict = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(state_dict)

    elif model_type == 'ocgan':
        enc = get_encoder().to(device)
        dec = get_decoder().to(device)

        checkpoint_en = torch.load(saved_model_path[0])
        enc.load_state_dict(checkpoint_en['state_dict'])

        checkpoint_de = torch.load(saved_model_path[1])
        dec.load_state_dict(checkpoint_de['state_dict'])
        model = [enc, dec]
    elif model_type == 'gan':
        model = torch.load(saved_model_path)
        model.to(device)

    else:
        model = None

    test(model, data_folders, model_type, num_frames=96, num_move_frames=96)


def test(model, data_folders, model_type, num_frames, num_move_frames):
    """
    Test the given model on the data in the given data folders
        :param model: The model to be tested or [encoder_model, decoder_model] for OCGAN
        :param data_folders: The list of data folders to test the model on.
        :param model_type: type of model that will be tested. Can be densevae, vqvae, ocgan, gan
        :param num_frames: The number of frames to be used for testing the model.
        :param num_move_frames: The number of frames to move for the next test.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    all_real = []
    all_fake = []
    avg_fake = []
    max_fake = []
    min_fake = []
    std_fake = []
    avg_real = []
    max_real = []
    min_real = []
    std_real = []
    n_seq_real = []
    n_seq_fake = []
    search_text = 'original'

    criterion = nn.MSELoss()

    # create master.npz file if it doesn't exist
    for folder_path in data_folders:
        master_path = os.path.join(folder_path, 'master.npz')
        if not os.path.exists(master_path):
            print('Creating master.npz in path: ', folder_path)
            create_masters(folder_path, remove=False)

    for _, folder_path in enumerate(tqdm(data_folders)):
        rmses_video = list()
        start_frame = 0

        # load the pose data from master.npz file
        with np.load(os.path.join(folder_path, 'master.npz')) as data:
            pose = data['pose']
        num_video_frames = pose.shape[0]

        # check if the video is original or fake
        if folder_path.find(search_text) != -1:  # original video
            orig_video = True
        else:
            orig_video = False
        if model_type == 'ocgan':
            enc = model[0]
            dec = model[1]
            enc.eval()
            dec.eval()
        else:
            model.eval()

        with torch.no_grad():
            i = 0
            while start_frame + num_frames <= num_video_frames:
                i += 1

                # create input for the model
                data = create_input(folder_path, num_frames, attrs=['exp', 'pose', 'shape'], start_idx=start_frame)
                data = data.to(device)

                # get the reconstructed input
                if model_type == 'densevae':
                    recon, _, _ = model(data)
                elif model_type == 'vqvae':
                    _, recon, _ = model(data)
                elif model_type == 'ocgan':
                    recon = dec(enc(data))
                elif model_type == 'gan':
                    recon = model(data)
                else:
                    print('Wrong inputs')
                    exit()

                # find the RMSE between the original and the reconstructed input
                rmse = torch.sqrt(criterion(recon.squeeze(), data.squeeze()))
                rmses_video.append(rmse.item())
                start_frame += num_move_frames

        # store the results of the test to their respective files
        if orig_video:
            min_real.append(min(rmses_video))
            max_real.append(max(rmses_video))
            avg_real.append(sum(rmses_video) / len(rmses_video))
            std_real.append(np.std(rmses_video))
            all_real.extend(rmses_video)
            n_seq_real.append(i)
        else:
            min_fake.append(min(rmses_video))
            max_fake.append(max(rmses_video))
            avg_fake.append(sum(rmses_video) / len(rmses_video))
            std_fake.append(np.std(rmses_video))
            all_fake.extend(rmses_video)
            n_seq_fake.append(i)

    np.savez(
        './ForFF/real_n_between_fr_' + str(num_move_frames), min_real=min_real,
        avg_real=avg_real, max_real=max_real, std_real=std_real, all_real=all_real, n_seq_real=n_seq_real
        )
    np.savez(
        './ForFF/fake_n_between_fr_' + str(num_move_frames), min_fake=min_fake,
        avg_fake=avg_fake, max_fake=max_fake, std_fake=std_fake, all_fake=all_fake, n_seq_fake=n_seq_fake
        )


def main(mode: int, threshold_mode_1, saved_model_path, face_forensics_path, model_type):
    """
    Method for testing the trained model for FaceForensics++ datasets. Contains 3 modes.
        - Mode 0: test the pretrained model for the dataset and save results to ./ForFF/ folder
        - Mode 1: find the RMSE threshold to differentiate real from fake reconstructed data
        - Mode 2: using the given threshold, calculate metrics for the 3 different comparison types described in chapter
          4.3 of thesis
        :param mode: mode to execute
        :param threshold_mode_1: RMSE threshold used in mode 1. Reconstructions with RMSE value larger than the threshold
         are considered fake
        :param saved_model_path: path of the save model or list of paths in case the model is the ocgan
        :param face_forensics_path: path where the faceforensics++ dataset is located
        :param model_type: type of model that will be tested. Can be densevae, vqvae, ocgan, gan
    """

    if mode == 0:
        test_trained_model(
            saved_model_path, face_forensics_path, model_type)

    else:

        real_npz_path = './ForFF/real_n_between_fr_96.npz'
        fake_npz_path = './ForFF/fake_n_between_fr_96.npz'

        with np.load(real_npz_path) as data1:
            real = data1['all_real']
            num_real = data1['n_seq_real']
            avg_real = data1['avg_real']

        with np.load(fake_npz_path) as data2:
            fake = data2['all_fake']
            num_fake = data2['n_seq_fake']
            avg_fake = data2['avg_fake']

        if mode == 1:
            find_threshold(avg_real.tolist(), avg_fake.tolist())

        elif mode == 2:

            true_labels = []
            pred_labels = []

            idx_real = 0
            idx_fake = 0

            for num in num_real:
                true_labels.append(1)
                real_seg = 0
                fake_seg = 0
                for i in range(0, num):
                    if real[idx_real] > threshold_mode_1:
                        fake_seg += 1
                    else:
                        real_seg += 1
                    idx_real += 1
                if real_seg > fake_seg:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

            for num in num_fake:
                true_labels.append(0)
                real_seg = 0
                fake_seg = 0
                for i in range(0, num):
                    if fake[idx_fake] > threshold_mode_1:
                        fake_seg += 1
                    else:
                        real_seg += 1
                    idx_fake += 1
                if real_seg > fake_seg:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

            print(
                ' Method most sequences of video : accuracy {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f} auc-roc score {}\n'
                .format(
                    accuracy_score(true_labels, pred_labels),
                    precision_score(true_labels, pred_labels),
                    recall_score(true_labels, pred_labels),
                    f1_score(true_labels, pred_labels),
                    roc_auc_score(true_labels, pred_labels)
                    )
                )

            all_rmses_videos = np.concatenate((avg_real, avg_fake))
            preds = list()

            for vid in all_rmses_videos:
                if vid > threshold_mode_1:
                    preds.append(0)
                else:
                    preds.append(1)

            print(
                    ' Method average rmse video : accuracy {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f} auc-roc score {}\n'.format(
                            accuracy_score(true_labels, preds),
                            precision_score(true_labels, preds),
                            recall_score(true_labels, preds),
                            f1_score(true_labels, preds),
                            roc_auc_score(true_labels, preds)
                            )
                    )

            true_labels_all = list()
            for _ in real:
                true_labels_all.append(1)
            for _ in fake:
                true_labels_all.append(0)

            all_rmses_seq = np.concatenate((real, fake))

            preds_seq = list()

            for vid in all_rmses_seq:
                if vid > threshold_mode_1:
                    preds_seq.append(0)
                else:
                    preds_seq.append(1)

            print(
                    ' Method all Sequences : accuracy {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f} auc-roc score {:.4f}'.format(
                            accuracy_score(true_labels_all, preds_seq),
                            precision_score(true_labels_all, preds_seq),
                            recall_score(true_labels_all, preds_seq),
                            f1_score(true_labels_all, preds_seq),
                            roc_auc_score(true_labels_all, preds_seq)
                            )
                    )

            RocCurveDisplay.from_predictions(true_labels, preds)
            ConfusionMatrixDisplay.from_predictions(true_labels, preds)

            plt.show()


if __name__ == '__main__':

    main(mode=0,
         threshold_mode_1=-1,
         saved_model_path='./models/denseVAE.pt',
         face_forensics_path='./data/ff_folder/',
         model_type='densevae')

    # main(
    #     mode=0,
    #     threshold_mode_1=-1,
    #     saved_model_path='./models/VQVAE.pt',
    #     face_forensics_path='./data/ff_folder/',
    #     model_type='vqvae'
    #     )
    #
    # main(
    #     mode=0,
    #     threshold_mode_1=-1,
    #     saved_model_path=['./model/enc_model.pth.tar', './model/dec_model.pth.tar'],
    #     face_forensics_path='./data/ff_folder/',
    #     model_type='ocgan'
    #     )
    #
    # main(
    #     mode=0,
    #     threshold_mode_1=-1,
    #     saved_model_path='./models/gan.pth',
    #     face_forensics_path='./data/ff_folder/',
    #     model_type='gan'
    #     )
