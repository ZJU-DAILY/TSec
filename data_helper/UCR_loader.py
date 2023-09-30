import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data_helper.dataset_helper import get_dataset_info


def load_txt_file(datadir, dataset, suffix=False):
    fdir = os.path.join(datadir, dataset)
    assert os.path.isdir(fdir), f"{fdir}. {dataset} could not be found in {datadir}"
    # again, for file names
    f_name = os.path.join(fdir, dataset)

    if suffix is False:
        data_train = np.loadtxt(f_name + '_TRAIN', delimiter=',')
        data_test_val = np.loadtxt(f_name + '_TEST', delimiter=',')
    else:
        data_train = np.loadtxt(f_name + '_TRAIN.txt', delimiter=None)
        data_test_val = np.loadtxt(f_name + '_TEST.txt', delimiter=None)

    # get data
    X_train = data_train[:, 1:]
    X_test = data_test_val[:, 1:]
    # get labels (numerical, not one-hot encoded)
    y_train = data_train[:, 0]
    y_test = data_test_val[:, 0]

    return X_train, X_test, y_train, y_test


def processed_UCR_data(datadir, dataset, suffix=False):
    # load data in numpy format
    X_train, X_test, y_train, y_test = load_txt_file(datadir, dataset, suffix=suffix)
    # add a thrid channel for univariate data
    if len(X_train.shape) < 3:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

    # Fix labels (some UCR datasets have negative labels)
    class_names = np.unique(y_train, axis=0)
    y_train_tmp = np.zeros(len(y_train))
    y_test_tmp = np.zeros(len(y_test))
    for i, class_name in enumerate(class_names):
        y_train_tmp[y_train == class_name] = i
        y_test_tmp[y_test == class_name] = i

    # Fixed
    y_train = y_train_tmp
    y_test = y_test_tmp

    # Switch channel dim ()
    # Torch data format is  [N, C, W] W=timesteps
    X_train = np.swapaxes(X_train, 2, 1)
    X_test = np.swapaxes(X_test, 2, 1)

    return X_train, X_test, y_train, y_test


def np_to_dataloader(X, y, batch_size=32, shuffle=True):
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    y_tensor = y_tensor.long()

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return dataloader


def get_train_and_validation_loaders(dataloader, validation_split=0.1, batch_size=32, shuffle=True, rand_seed=42):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataloader.dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(rand_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataloader.dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataloader.dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, validation_loader


def get_UCR_data(datadir, dataset_name, batch_size=32):
    X_train, X_test, y_train, y_test = processed_UCR_data(datadir, dataset_name)
    input_shape, n_classes = get_dataset_info(dataset_name, X_train, X_test, y_train, y_test, print_info=True)

    train_dataloader = np_to_dataloader(X_train, y_train, batch_size, shuffle=True)
    train_dataloader, validation_dataloader = get_train_and_validation_loaders(train_dataloader,
                                                                               validation_split=0.1,
                                                                               batch_size=batch_size)

    test_dataloader = np_to_dataloader(X_test, y_test, batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader
