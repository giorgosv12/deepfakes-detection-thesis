from One_Class_Classifiers.VAEs.models.VQVAE import VQVae
from Utils.dataset import Vox_dataset
from One_Class_Classifiers.VAEs.models.DenseVAE import Dense_VAE, customLoss
from tqdm import tqdm
import torch
from torch import optim
import numpy as np
import random
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F


def train_dense_VAE(model_is_denseVAE=False, model_is_VQVAE=False):
    """
    This method is responsible for training the model. Model can be either DenseVAE or VQ-VAEs
        :param model_is_denseVAE: whether the model is a DenseVAE (default: False)
        :param model_is_VQVAE: whether the model is a VQVAE (default: False)
    """

    # inputs validation
    if not model_is_VQVAE != model_is_denseVAE:
        print('Only one option model_is_denseVAE or model_is_VQVAE can be True')
        exit()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(42)

    # percentage of train data
    train_perc = 0.9

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

    # get the paths of the folders containing the transformed master.npz files and shuffle them
    data_folders_paths = np.load('./data_folders_paths.npy').tolist()
    random.shuffle(data_folders_paths)

    # split the paths of the files into train and test sets
    train_paths = data_folders_paths[:int(train_perc * len(data_folders_paths))]
    test_paths = data_folders_paths[int(train_perc * len(data_folders_paths)):]

    # initialize model
    if model_is_denseVAE:
        model = Dense_VAE(latent_dims).to(device)
    elif model_is_VQVAE:
        model = VQVae(num_hiddens=128, num_residual_layers=6, num_residual_hiddens=32,
                      num_embeddings=512, embedding_dim=64,
                      commitment_cost=0.25).to(device)
    else:
        model = None

    # create train and test datasets and their respective dataloaders
    train_dataset = Vox_dataset(train_paths, num_frames, create_masters=False, skip_checks=False, train_for_all_data=False)
    test_dataset = Vox_dataset(test_paths, num_frames, create_masters=False, skip_checks=False, train_for_all_data=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=2 * 1e-4)
    closs = customLoss()
    train_losses = []
    val_losses = []
    best_val_loss = 50000.0
    epochs_no_change = 0

    mse_loss = nn.MSELoss()

    # training process
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = 0

        for data in train_dataloader:
            data = data.to(device)

            if model_is_denseVAE:
                recon, mu, logvar = model(data)
                loss = closs(recon, data, mu, logvar)
            elif model_is_VQVAE:
                vq_loss, recon, perplexity = model(data)
                vq_loss = vq_loss.mean()
                recon_error = F.mse_loss(data, recon)
                loss = (recon_error + vq_loss)
            else:
                loss = None

            for param in model.parameters():
                param.grad = None
            loss.backward()
            train_loss += loss.item()

            clip_value = 5
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

        # evaluation phase
        model.eval()
        val_loss = 0
        mse_losses = 0
        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(device)

                if model_is_denseVAE:
                    recon, mu, logvar = model(data)
                    loss = closs(recon, data, mu, logvar)
                    mse = mse_loss(recon, data)

                elif model_is_VQVAE:
                    vq_loss, recon, perplexity = model(data)
                    vq_loss = vq_loss.mean()
                    recon_error = F.mse_loss(recon, data)
                    mse = mse_loss(data, recon)
                    loss = (recon_error + vq_loss)

                mse_losses += mse.item()
                val_loss += loss.item()

        print(' Ep: {} Train loss: {:.6f},  Val Loss: {:.6f}, L1:{:.6f}'.format(
                    epoch, train_loss / len(train_dataloader.dataset), val_loss / len(test_dataloader.dataset),
                           mse_losses / len(test_dataloader.dataset)))

        train_losses.append(train_loss / len(train_dataloader.dataset))
        val_losses.append(val_loss / len(test_dataloader.dataset))

        if epoch >= mand_epochs:
            if val_loss / len(test_dataloader.dataset) >= best_val_loss:
                epochs_no_change += 1
            else:
                epochs_no_change = 0
                best_val_loss = val_loss / len(test_dataloader.dataset)
                torch.save(model.state_dict(), './model/best_model.pt')

            if epochs_no_change >= patience:
                torch.save(model.state_dict(), './model/model_early' + str(epoch) + '.pt')
                np.save('./model/train_losses', train_losses)
                np.save('./model/val_losses', val_losses)
                exit(epoch)

    np.save('./model/train_losses', train_losses)
    np.save('./model/val_losses', val_losses)


if __name__ == '__main__':
    train_dense_VAE()
