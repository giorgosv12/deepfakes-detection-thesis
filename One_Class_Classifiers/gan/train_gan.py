import torch
import numpy as np
import random
import os
from tqdm import tqdm
from Utils.dataset import Vox_dataset
from model import Generator_Net, Discriminator_Net, R_Loss, D_Loss


def main():
    """
    Main function to run the training process.
    """
    # Load the data folder paths
    data_folders_paths = np.load('./data_folders_paths.npy').tolist()

    # Shuffle the data folder paths
    random.shuffle(data_folders_paths)

    # Set the percentage of data for training
    train_perc = 0.9

    # Set the number of frames
    num_frames = 96

    # Split the data folder paths into training and testing sets
    train_paths = data_folders_paths[:int(train_perc * len(data_folders_paths))]
    test_paths = data_folders_paths[int(train_perc * len(data_folders_paths)):]

    # Create training and validation datasets
    train_dataset = Vox_dataset(train_paths, num_frames, create_masters=False, skip_checks=True,
                                train_for_all_data=False)
    valid_dataset = Vox_dataset(test_paths, num_frames, create_masters=False, skip_checks=True,
                                train_for_all_data=False)

    # Set device for training
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize the generator network
    gen_net = Generator_Net().to(device)

    # Initialize the discriminator network
    dis_net = Discriminator_Net(in_resolution=(96, 156)).to(device)

    # Set training parameters
    save_path = ('./', 'g_net.pth', 'd_net.pth')
    optim_r_params = {'alpha': 0.9, 'weight_decay': 1e-9}
    optim_d_params = {'alpha': 0.9, 'weight_decay': 1e-9}

    # Train the model
    train_model(gen_net, dis_net, train_dataset, valid_dataset, R_Loss, D_Loss, optimizer_class=torch.optim.RMSprop,
                device=device, batch_size=900, optim_r_params=optim_r_params, optim_d_params=optim_d_params,
                learning_rate=0.0002, rec_loss_bound=0.005,
                save_step=4, num_workers=4, save_path=save_path, lambd=0.2)


def train_model(g_net: torch.nn.Module,
                d_net: torch.nn.Module,
                train_dataset: torch.utils.data.Dataset,
                valid_dataset: torch.utils.data.Dataset,
                r_loss,
                d_loss,
                lr_scheduler = True,
                optimizer_class = torch.optim.RMSprop,
                optim_r_params: dict = {},
                optim_d_params: dict = {},
                learning_rate: float = 0.0002,
                batch_size: int = 512,
                pin_memory: bool = True,
                num_workers: int = 6,
                max_epochs: int = 2000,
                epoch_step: int = 1,
                save_step: int = 5,
                rec_loss_bound: float = 0.1,
                lambd: float = 0.2,
                device: torch.device = torch.device('cuda:0'),
                save_path: tuple = ('.','g_net','d_net')):
    """
    Train a GAN model on a given dataset.
        :param g_net: generator network
        :param d_net: discriminator network
        :param train_dataset: training dataset
        :param valid_dataset: validation dataset
        :param r_loss: reconstruction loss function
        :param d_loss: discriminator loss function
        :param lr_scheduler: flag indicating if lr_scheduler will be used (default=True)
        :param optimizer_class: optimizer class to use (default=torch.optim.RMSprop)
        :param optim_r_params: additional parameters to be passed to generator optimizer
        :param optim_d_params: additional parameters to be passed to discriminator optimizer
        :param learning_rate: learning rate for optimizers (default=0.0002)
        :param batch_size: batch size for training (default=512)
        :param pin_memory: flag indicating if pin memory will be used for data loaders (default=True)
        :param num_workers: number of workers for data loaders (default=6)
        :param max_epochs: maximum number of epochs to train (default=2000)
        :param epoch_step: epoch interval to log progress (default=1)
        :param save_step: epoch interval to save model (default=5)
        :param rec_loss_bound: stop the training process if this recreation error is reached (default=0.1)
        :param lambd: scalar value for weighting between reconstruction loss and generator loss (deafult=0.2)
        :param device: device where training will be executed (default=cuda:0)
        :param save_path: save paths outputs and trained models (default= ('.','g_net','d_net'))
    """

    model_path = os.path.join(save_path[0], 'models')
    metric_path = os.path.join(save_path[0], 'metrics')
    r_net_path = os.path.join(model_path, save_path[1])
    d_net_path = os.path.join(model_path, save_path[2])

    # create output folders
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)

    print(f'Models will be saved in {r_net_path} and {d_net_path}')
    print(f'Metrics will be saved in {metric_path}')

    optim_r = optimizer_class(g_net.parameters(), lr = learning_rate, **optim_r_params)
    optim_d = optimizer_class(d_net.parameters(), lr = learning_rate, **optim_d_params)

    # create learning rate scheduler if chosen
    if lr_scheduler:
        print('Using Scheduler')
        scheduler_r = torch.optim.lr_scheduler.StepLR(optim_r, step_size=25, gamma=0.2)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, step_size=25, gamma=0.2)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=pin_memory,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, pin_memory=pin_memory,
                                               num_workers=num_workers)

    metrics =  {'train' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : []},
                'valid' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : []}}

    # for each epoch
    for epoch in tqdm(range(max_epochs)):

        train_metrics = train_single_epoch(g_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device)
        valid_metrics = validate_single_epoch(g_net, d_net, r_loss, d_loss, valid_loader, device)

        metrics['train']['rec_loss'].append(train_metrics['rec_loss'])
        metrics['train']['gen_loss'].append(train_metrics['gen_loss'])
        metrics['train']['dis_loss'].append(train_metrics['dis_loss'])
        metrics['valid']['rec_loss'].append(valid_metrics['rec_loss'])
        metrics['valid']['gen_loss'].append(valid_metrics['gen_loss'])
        metrics['valid']['dis_loss'].append(valid_metrics['dis_loss'])

        if epoch % epoch_step == 0:
            print(f'Epoch {epoch}:')
            print('TRAIN METRICS:', train_metrics)
            print('VALID METRICS:', valid_metrics)

        if lr_scheduler:
            scheduler_r.step()
            scheduler_d.step()

        if epoch % save_step == 0:
            torch.save(g_net, r_net_path+str(epoch)+'.pth')
            torch.save(d_net, d_net_path+str(epoch)+'.pth')
            print(f'Saving model on epoch {epoch}')

        if valid_metrics['rec_loss'] < rec_loss_bound and train_metrics['rec_loss'] < rec_loss_bound:
            torch.save(g_net, r_net_path)
            torch.save(d_net, d_net_path)
            print('Stopping training')
            break


def train_single_epoch(g_net, d_net, optim_r, optim_d, g_loss, d_loss, train_loader, lambd, device) -> dict:
    """
    Trains the generator and discriminator for a single epoch.
        :param g_net: The generator network.
        :param d_net: The discriminator network.
        :param optim_r: Optimizer for the generator network.
        :param optim_d: Optimizer for the discriminator network.
        :param g_loss: Loss function for the generator network.
        :param d_loss: Loss function for the discriminator network.
        :param train_loader: The data loader for the training set.
        :param lambd: The lambda value in the calculation of the generator loss.
        :param device: The device (CPU or GPU) to use for training.
        :return: A dictionary of average losses for the generator (gen_loss), discriminator (dis_loss), and reconstruction
            loss (rec_loss) for this epoch.
    """
    g_net.train()
    d_net.train()

    train_metrics = {'rec_loss': 0, 'gen_loss': 0, 'dis_loss': 0}

    for data in train_loader:
        x_real = data.to(device)
        x_fake = g_net(x_real)

        d_net.zero_grad()
        dis_loss = d_loss(d_net, x_real, x_fake)
        dis_loss.backward()
        optim_d.step()
        g_net.zero_grad()

        r_metrics = g_loss(d_net, x_real, x_fake, lambd)  # L_r = gen_loss + lambda * rec_loss
        r_metrics['L_r'].backward()
        optim_r.step()

        train_metrics['rec_loss'] += r_metrics['rec_loss']
        train_metrics['gen_loss'] += r_metrics['gen_loss']
        train_metrics['dis_loss'] += dis_loss

    train_metrics['rec_loss'] = train_metrics['rec_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)
    train_metrics['gen_loss'] = train_metrics['gen_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)
    train_metrics['dis_loss'] = train_metrics['dis_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)

    return train_metrics


def validate_single_epoch(g_net, d_net, g_loss, d_loss, valid_loader, device) -> dict:
    """
    Validates the trained model for a single epoch.
        :param g_net: The generator network.
        :param d_net: The discriminator network.
        :param g_loss: Loss function for the generator network.
        :param d_loss: Loss function for the discriminator network.
        :param valid_loader: The data loader for the validation set.
        :param device: The device (CPU or GPU) to use for training.
        :return: A dictionary of average losses for the generator (gen_loss), discriminator (dis_loss), and reconstruction
            loss (rec_loss) for this epoch.
        """

    g_net.eval()
    d_net.eval()

    valid_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0}

    with torch.no_grad():
        for data in valid_loader:

            x_real = data.to(device)
            x_fake = g_net(x_real)

            dis_loss = d_loss(d_net, x_real, x_fake)

            r_metrics = g_loss(d_net, x_real, x_fake, 0)

            valid_metrics['rec_loss'] += r_metrics['rec_loss']
            valid_metrics['gen_loss'] += r_metrics['gen_loss']
            valid_metrics['dis_loss'] += dis_loss

    valid_metrics['rec_loss'] = valid_metrics['rec_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
    valid_metrics['gen_loss'] = valid_metrics['gen_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
    valid_metrics['dis_loss'] = valid_metrics['dis_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)

    return valid_metrics


if __name__ == "__main__":
    main()
