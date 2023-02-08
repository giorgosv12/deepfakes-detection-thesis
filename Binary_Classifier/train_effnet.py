from random import seed
from dataset import Vox_dataset
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn, optim

from Utils.dataset import Vox_dataset
from Binary_Classifier.model.effnetv2 import effnetv2_s


def train_binary_model():
    """
    This function trains a binary classification model using the VoxCeleb2 and the FaceForensics++ datasets.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    seed(42)

    # number of sequential frames that will be used for each video
    num_frames = 96

    # batch size
    batch_size = 160

    # number of epochs to train
    epochs = 300

    # number of dimensions for latent representation
    latent_dims = 2000

    # number of minimum epochs
    mand_epochs = 100

    # number of epochs without change before early stopping ends the training
    patience = 20

    # folder where model will be saved
    savefolder = './model/'

    # model initialization
    model = effnetv2_s().to(device)

    # paths of the files that will be used for training
    train_paths = np.load('./data/train_paths.npy')

    # paths of the files that will be used for testing
    test_paths = np.load('./data/test_paths.npy')

    target = list()
    for path in train_paths:
        if path.find('vox') != -1:
            target.append(1)
        else:
            target.append(0)

    target = np.array(target)
    print(
            'target train 0/1: {}/{}'.format(
                    len(np.where(target == 0)[0]), len(np.where(target == 1)[0])
                    )
            )

    class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)]
            )

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    # create a Weight Sampler as the number of samples of the two classes have a large difference
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # create the train and test datasets and their dataloaders
    train_dataset = Vox_dataset(
        train_paths, num_frames, create_masters=False, delete_masters=False, skip_checks=True,
        train_for_all_data=False, return_label=True, search_text_original='results_voxceleb2'
        )
    test_dataset = Vox_dataset(
        test_paths, num_frames, create_masters=False, delete_masters=False, skip_checks=True,
        train_for_all_data=False, return_label=True, search_text_original='results_voxceleb2'
        )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # adam optimizer used for model training
    optimizer = optim.Adam(model.parameters(), lr=0.5 * 1e-3)

    train_losses = []
    val_losses = []
    best_val_loss = 50000.0
    epochs_no_change = 0

    # binary cross entropy will be used as a loss
    closs = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(0, epochs + 1)):
        model.train()
        train_loss = 0

        for label, data in train_dataloader:

            label = label.to(device)
            data = data.float().to(device)

            predicted = model(data).squeeze()

            loss = closs(predicted, label.float())
            for param in model.parameters():
                param.grad = None
            loss.backward()
            train_loss += loss.item()

            clip_value = 5
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for label, data in test_dataloader:
                label = label.to(device)
                data = data.float().to(device)

                predicted = model(data).squeeze()
                loss = closs(predicted, label.float())

                val_loss += loss.item()

        print(
                ' Ep: {} Train loss: {:.6f},  Val Loss: {:.6f}'.format(
                        epoch, train_loss / len(train_dataloader.dataset), val_loss / len(test_dataloader.dataset),
                        )
                )

        train_losses.append(train_loss / len(train_dataloader.dataset))
        val_losses.append(val_loss / len(test_dataloader.dataset))

        if epoch >= mand_epochs:
            if val_loss / len(test_dataloader.dataset) >= best_val_loss:
                epochs_no_change += 1
            else:
                epochs_no_change = 0
                best_val_loss = val_loss / len(test_dataloader.dataset)
                torch.save(model.state_dict(), savefolder + 'best_model.pt')

            if epochs_no_change >= patience:
                torch.save(model.state_dict(), savefolder + 'model_early' + str(epoch) + '.pt')
                np.save(savefolder + 'train_losses', train_losses)
                np.save(savefolder + 'val_losses', val_losses)

    np.save(savefolder + 'train_losses', train_losses)
    np.save(savefolder + 'val_losses', val_losses)


if __name__ == '__main__':
    train_binary_model()
