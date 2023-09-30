import math
import os

import numpy as np
import weka.core.converters
import weka.core.jvm

from data_helper.UCR_loader import np_to_dataloader, get_train_and_validation_loaders
from data_helper.dataset_helper import get_dataset_info


def load_UEA_dataset(path, dataset):
    weka.core.jvm.start()
    loader = weka.core.converters.Loader(
        classname="weka.core.converters.ArffLoader"
    )

    train_file = os.path.join(path, dataset, dataset + "_TRAIN.arff")
    test_file = os.path.join(path, dataset, dataset + "_TEST.arff")
    train_weka = loader.load_file(train_file)
    test_weka = loader.load_file(test_file)

    train_size = train_weka.num_instances
    test_size = test_weka.num_instances
    nb_dims = train_weka.get_instance(0).get_relational_value(0).num_instances
    length = train_weka.get_instance(0).get_relational_value(0).num_attributes

    train = np.empty((train_size, nb_dims, length))
    test = np.empty((test_size, nb_dims, length))

    train_labels = np.empty(train_size, dtype=int)
    test_labels = np.empty(test_size, dtype=int)

    for i in range(train_size):
        train_labels[i] = int(train_weka.get_instance(i).get_value(1))
        time_series = train_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            train[i, j] = time_series.get_instance(j).values

    for i in range(test_size):
        test_labels[i] = int(test_weka.get_instance(i).get_value(1))
        time_series = test_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            test[i, j] = time_series.get_instance(j).values

    # Normalizing dimensions independently
    for j in range(nb_dims):
        mean = np.mean(np.concatenate([train[:, j], test[:, j]]))
        var = np.var(np.concatenate([train[:, j], test[:, j]]))
        train[:, j] = (train[:, j] - mean) / math.sqrt(var)
        test[:, j] = (test[:, j] - mean) / math.sqrt(var)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    train_labels = np.vectorize(transform.get)(train_labels)
    test_labels = np.vectorize(transform.get)(test_labels)

    weka.core.jvm.stop()
    print('dataset load succeed !!!')
    return train, train_labels, test, test_labels


def processed_UEA_data(path, dataset):
    X_train, y_train, X_test, y_test = load_UEA_dataset(path, dataset)

    if len(X_train.shape) < 3:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

    # Fix labels
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
    # fixme: 因为使用weka导入，默认数据格式已经是[N,C,W]
    # X_train = np.swapaxes(X_train, 2, 1)
    # X_test = np.swapaxes(X_test, 2, 1)

    return X_train, X_test, y_train, y_test


def get_UEA_data(path, dataset_name, batch_size=32):
    X_train, X_test, y_train, y_test = processed_UEA_data(path, dataset_name)
    input_shape, n_classes = get_dataset_info(dataset_name, X_train, X_test, y_train, y_test, print_info=True)
    train_dataloader = np_to_dataloader(X_train, y_train, batch_size, shuffle=True)
    train_dataloader, validation_dataloader = get_train_and_validation_loaders(train_dataloader, validation_split=0.1,
                                                                               batch_size=batch_size)
    test_dataloader = np_to_dataloader(X_test, y_test, batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # train, test, train_labels, test_labels = processed_UEA_data('../examples/data', 'Libras')
    # train_dataloader, validation_dataloader, test_dataloader = get_UEA_data('../examples/data', 'Libras')
    print('---end---')
