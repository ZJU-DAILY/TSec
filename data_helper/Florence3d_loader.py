import os

import numpy as np

from data_helper.UCR_loader import np_to_dataloader, get_train_and_validation_loaders
from data_helper.dataset_helper import get_dataset_info


# 10 people 9 labels, use this information to separate data manually
def load_Florence3d_dataset(path, dataset_name='Florence3d'):
    fdir = os.path.join(path, dataset_name)
    assert os.path.isdir(fdir), f"in {path}, can not find {dataset_name}"
    f_name = os.path.join(fdir, dataset_name)

    total_data = np.loadtxt(f_name + ".txt")
    total_data = total_data[:, 1:]
    data_train = []
    data_test = []
    for people in range(1, 11):
        people_idx = total_data[:, 0] == people
        people_data = total_data[people_idx]
        for action in range(1, 10):
            action_idx = people_data[:, 1] == action
            people_action_data = people_data[action_idx]
            split = int(np.ceil(people_action_data.shape[0] * 0.5))
            data_train.extend(people_action_data[:split, :].tolist())
            data_test.extend(people_action_data[split:, :].tolist())
    np_train = np.array(data_train)
    np_test = np.array(data_test)
    X_train = np_train[:, 2:]
    y_train = np_train[:, 1]
    X_test = np_test[:, 2:]
    y_test = np_test[:, 1]

    X_train = np.reshape(X_train, (-1, 15, 3))
    X_test = np.reshape(X_test, (-1, 15, 3))

    return X_train, X_test, y_train, y_test


def get_Florence3d_data(path, dataset_name='Florence3d', batch_size=32):
    X_train, X_test, y_train, y_test = load_Florence3d_dataset(path, dataset_name)
    input_shape, n_classes = get_dataset_info(dataset_name, X_train, X_test, y_train, y_test, print_info=True)
    train_dataloader = np_to_dataloader(X_train, y_train, batch_size, shuffle=True)
    train_dataloader, validation_dataloader = get_train_and_validation_loaders(train_dataloader, validation_split=0.1,
                                                                               batch_size=batch_size)

    test_dataloader = np_to_dataloader(X_test, y_test, batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader


if __name__ == '__main__':
    # load_Florence3d_dataset('../examples/data')
    get_Florence3d_data('../examples/data')
