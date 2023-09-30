import argparse

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from CFDTAN.CFDTAN_layer import CfDtAn
from CFDTAN.alignment_loss import alignment_loss, alignment_loss_T
from utils.utils_func import CFArgs, ExperimentClass


def train_epoch(train_loader, device, optimizer, model, channels, cf_args):
    train_loss = 0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output, thetas = model(data, return_theta=True)

        loss = alignment_loss(output, target, thetas, channels, cf_args)
        train_loss += loss
        loss.backward()
        optimizer.step()

    return train_loss


def train_epoch_T(train_loader, device, optimizer, model, channels, cf_args, T):
    train_loss = 0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output, thetas = model(data, return_theta=True)

        loss = alignment_loss_T(output, target, thetas, channels, cf_args, T)
        train_loss += loss
        loss.backward()
        optimizer.step()

    return train_loss


def validation_epoch(val_loader, device, model, channels, cf_args):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        # prior_loss = 0
        # align_loss = 0

        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, theta = model(data, return_theta=True)

            val_loss += alignment_loss(output, target, theta, channels, cf_args)

        return val_loss


def validation_epoch_T(val_loader, device, model, channels, cf_args, T):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        # prior_loss = 0
        # align_loss = 0

        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, theta = model(data, return_theta=True)

            val_loss += alignment_loss_T(output, target, theta, channels, cf_args, T)

        return val_loss


def _save_checkpoint(model, optimizer, test_loss, exp_name=''):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': test_loss
    }

    torch.save(checkpoint, f'../checkpoints/{exp_name}_checkpoint.pth')


def _save_checkpoint_T(model, optimizer, test_loss, exp_name=''):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': test_loss
    }

    torch.save(checkpoint, f'./checkpoints/{exp_name}_checkpoint.pth')


def train(train_loader, val_loader, cf_args: CFArgs, experiment: ExperimentClass, print_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    channels, input_shape = train_loader.dataset[0][0].shape

    model = CfDtAn(input_shape, channels, tess_size=cf_args.tess_size, n_recur=cf_args.n_recurrences,
                   zero_boundary=cf_args.zero_boundary, device='gpu', num_scaling=cf_args.n_ss,
                   back_version=cf_args.back_version).to(device)
    cf_args.T = model.get_basis()
    optimizer = optim.Adam(model.parameters(), lr=experiment.lr)

    if print_model:
        print(model)
        print(cf_args)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('# parameters:', pytorch_total_params)

    min_loss = np.inf
    for epoch in tqdm(range(1, experiment.n_epochs + 1)):
        train_loss = train_epoch(train_loader, device, optimizer, model, channels, cf_args)
        val_loss = validation_epoch(val_loader, device, model, channels, cf_args)
        if val_loss < min_loss:
            min_loss = val_loss
            _save_checkpoint(model, optimizer, val_loss, experiment.exp_name)
        if epoch % 50 == 0:
            train_loss /= len(train_loader.dataset)
            print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))
            val_loss /= len(val_loader.dataset)
            print('Validation set: Average loss: {:.4f}\n'.format(val_loss))

    checkpoint = torch.load(f'../checkpoints/{experiment.exp_name}_checkpoint.pth')

    return model


def train_T(train_loader, val_loader, cf_args: argparse.Namespace, print_model=False, suffix='', is_multi=False,
            save_model=True, betas=(0.9, 0.999)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('device', device)
    channels, input_shape = train_loader.dataset[0][0].shape

    model = CfDtAn(input_shape, channels, tess_size=cf_args.tess_size, n_recur=cf_args.n_recurrences,
                   zero_boundary=cf_args.zero_boundary, device='gpu', num_scaling=cf_args.n_ss,
                   back_version=cf_args.back_version).to(device)
    T = model.get_basis()
    optimizer = optim.Adam(model.parameters(), lr=cf_args.cf_lr, betas=betas)

    if print_model:
        print(model)
        print(cf_args)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('# parameters:', pytorch_total_params)

    min_loss = np.inf
    for epoch in tqdm(range(1, cf_args.cf_n_epochs + 1)):
        train_loss = train_epoch_T(train_loader, device, optimizer, model, channels, cf_args, T)
        val_loss = validation_epoch_T(val_loader, device, model, channels, cf_args, T)
        if save_model:
            if val_loss < min_loss:
                min_loss = val_loss
                # _save_checkpoint(model, optimizer, val_loss, experiment.exp_name)
                # _save_checkpoint_T(model, optimizer, val_loss, suffix)
                if is_multi:
                    torch.save(model.state_dict(), f'./checkpoints/alignment/multiple/{suffix}_checkpoint.pth')
                else:
                    torch.save(model.state_dict(), f'./checkpoints/alignment/single/{suffix}_checkpoint.pth')
                # torch.save(model, f'./checkpoints/{suffix}_checkpoint.pth')
        if epoch % 10 == 0:
            train_loss /= len(train_loader.dataset)
            print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))
            val_loss /= len(val_loader.dataset)
            print('Validation set: Average loss: {:.4f}\n'.format(val_loss))

    # checkpoint = torch.load(f'../checkpoints/{experiment.exp_name}_checkpoint.pth')
    # checkpoint = torch.load(f'./checkpoints/{suffix}_checkpoint.pth')

    return model
