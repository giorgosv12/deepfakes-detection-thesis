import os
from tqdm import tqdm
import numpy as np
import random
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, \
    RocCurveDisplay, ConfusionMatrixDisplay

from Binary_Classifier.model.effnetv2 import effnetv2_s
from Utils.create_tensor_from_npz import create_input


def test_trained_effnet(savefolder, num_frames, features, mode):
    """
    Method for testing the trained model for FaceForensics++ and VoxCeleb2 datasets. Contains 2 modes.
        - Mode 0: test the pretrained model for the dataset and save results
        - Mode 1:  calculate metrics for the 3 different comparison types described in chapter 4.3 of thesis
        :param mode: mode to execute
        :param savefolder: path of the save model and the saved files of mode 0
        :param num_frames: number of frame the model's input contains
        :param features: the features that the model's input contains
        """

    if mode == 0:

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        random.seed(42)

        # load the trained model
        model = effnetv2_s().to(device)
        state_dict = torch.load(savefolder + 'best_model.pt', map_location=device)
        model.load_state_dict(state_dict)

        # calculate scores for the test data
        test_paths = np.load('./data/test_paths.npy')
        test(model, test_paths, num_frames=num_frames, savefolder=savefolder, features=features)

    elif mode == 1:

        # the result files of mode 0
        real_npz_path = savefolder + 'true.npz'
        fake_npz_path = savefolder + 'false.npz'

        with np.load(real_npz_path) as data1:
            real = data1['all_real']
            num_real = data1['n_seq_real']
            avg_real = data1['avg_real']

        with np.load(fake_npz_path) as data2:
            fake = data2['all_fake']
            num_fake = data2['n_seq_fake']
            avg_fake = data2['avg_fake']

        true_labels = []
        pred_labels = []

        idx_real = 0
        idx_fake = 0

        for num in num_real:
            true_labels.append(1)
            real_seg = 0
            fake_seg = 0
            for i in range(0, num):
                if real[idx_real] < .5:
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
                if fake[idx_fake] < .5:
                    fake_seg += 1
                else:
                    real_seg += 1
                idx_fake += 1
            if real_seg > fake_seg:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        print(
            ' Method most sequences of video : accuracy {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f} auc-roc score {:.4f}\n'.format(
                accuracy_score(true_labels, pred_labels),
                precision_score(true_labels, pred_labels),
                recall_score(true_labels, pred_labels),
                f1_score(true_labels, pred_labels),
                roc_auc_score(true_labels, pred_labels)))

        all_scores_videos = np.concatenate((avg_real, avg_fake))
        preds = list()

        for vid in all_scores_videos:
            if vid < .5:
                preds.append(0)
            else:
                preds.append(1)

        print(
            ' Method average score video : accuracy {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f} auc-roc score {:.4f}\n'.format(
                accuracy_score(true_labels, preds),
                precision_score(true_labels, preds),
                recall_score(true_labels, preds),
                f1_score(true_labels, preds),
                roc_auc_score(true_labels, preds)))

        true_labels_all = list()
        for _ in real:
            true_labels_all.append(1)
        for _ in fake:
            true_labels_all.append(0)

        all_scores_seq = np.concatenate((real, fake))

        preds_seq = list()

        for vid in all_scores_seq:
            if vid < .5:
                preds_seq.append(0)
            else:
                preds_seq.append(1)

        print(
            ' Method all Sequences : accuracy {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f} auc-roc score {:.4f}'.format(
                accuracy_score(true_labels_all, preds_seq),
                precision_score(true_labels_all, preds_seq),
                recall_score(true_labels_all, preds_seq),
                f1_score(true_labels_all, preds_seq),
                roc_auc_score(true_labels_all, preds_seq)))

        RocCurveDisplay.from_predictions(true_labels, preds)
        ConfusionMatrixDisplay.from_predictions(true_labels, preds, values_format='')


def test(model, data_folders, num_frames, savefolder, features):
    """
    Evaluate a trained model to test data and save files containing the scores.
    :param model: the trained model to test
    :param data_folders: list containing the paths of folders with test data
    :param num_frames: number of frames of the model's input
    :param savefolder: folder where calculated scores will be saved
    :param features: feautres of the model's input
    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()

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

    # Loop over each folder path
    for _, folder_path in enumerate(tqdm(data_folders)):
        scores_video = list()
        start_frame = 0
        with np.load(os.path.join(folder_path, 'master.npz')) as data:
            pose = data['pose']
        num_video_frames = pose.shape[0]

        # check if the video is original or fake
        if folder_path.find('vox') != -1:  # original video
            orig_video = True
        else:
            orig_video = False

        with torch.no_grad():
            i = 0
            while start_frame + num_frames <= num_video_frames:
                i += 1

                # create input for the model
                data = create_input(folder_path, num_frames, attrs=features, start_idx=start_frame)
                data = data.float().to(device)

                # get the model's output score
                score = model(data)
                scores_video.append(expit(score.item()))
                start_frame += num_frames

        # store the results of the test to their respective files
        if orig_video:
            min_real.append(min(scores_video))
            max_real.append(max(scores_video))
            avg_real.append(sum(scores_video) / len(scores_video))
            std_real.append(np.std(scores_video))
            all_real.extend(scores_video)
            n_seq_real.append(i)
        else:
            min_fake.append(min(scores_video))
            max_fake.append(max(scores_video))
            avg_fake.append(sum(scores_video) / len(scores_video))
            std_fake.append(np.std(scores_video))
            all_fake.extend(scores_video)
            n_seq_fake.append(i)

    np.savez(savefolder + 'true.npz', min_real=min_real,
             avg_real=avg_real, max_real=max_real, std_real=std_real, all_real=all_real, n_seq_real=n_seq_real)
    np.savez(savefolder + 'false.npz', min_fake=min_fake,
             avg_fake=avg_fake, max_fake=max_fake, std_fake=std_fake, all_fake=all_fake, n_seq_fake=n_seq_fake)


if __name__ == '__main__':

    savefolder = './model/'

    # features that models were trained on. Three different subsets were used
    features1 = ['exp', 'pose', 'shape']
    features2 = ['exp', 'pose', 'shape', 'tex']
    features3 = ['exp', 'pose', 'shape', 'tex', 'angles', 'mocs', 'left_eye', 'right_eye']

    # choose the right subset of features for the trained model
    features = features1

    # number of frames that model was trained on. Models were trained for 50 and 96 frames
    num_frames = 96

    mode = 1  # 0 is create .npz, 1 is show results

    test_trained_effnet(savefolder, num_frames, features, mode)
