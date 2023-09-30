from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from data_helper.UCR_loader import processed_UCR_data


def plot_mean_signal_multi_one_by_one(X_aligned_within_class, X_within_class, ratio, class_num, dataset_name, N=10):
    # check data dim
    import os
    figure_father_path = './figures'
    figure_path = os.path.join(figure_father_path, dataset_name)
    print(figure_path)
    os.makedirs(figure_path, exist_ok=True)

    predefined_color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf']

    class_num = int(class_num)

    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    # data dims: (number of samples, dim, channels)
    n_signals = len(X_within_class)  # number of samples within class

    # Sample random signals
    input_shape = X_within_class.shape[1:]  # (channels, dims) PyTorch
    signal_len = input_shape[1]
    n_channels = input_shape[0]

    N = n_signals if N > n_signals else N
    indices = np.random.choice(n_signals, N, replace=False)  # N samples
    X_within_class = X_within_class[indices, :, :]  # get N samples, all channels
    X_aligned_within_class = X_aligned_within_class[indices, :, :]

    # Compute mean signal and variance
    X_mean_t = np.mean(X_aligned_within_class, axis=0)
    X_std_t = np.std(X_aligned_within_class, axis=0)
    upper_t = X_mean_t + X_std_t
    lower_t = X_mean_t - X_std_t

    X_mean = np.mean(X_within_class, axis=0)
    X_std = np.std(X_within_class, axis=0)
    upper = X_mean + X_std
    lower = X_mean - X_std

    # set figure size
    [w, h] = ratio  # width, height

    title_font = 22
    # plot each channel

    signal_len = int(signal_len / 4)

    t = range(input_shape[1])
    # Misaligned Signals
    f1, ax1 = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    f1.set_size_inches(w, h)
    for channel in range(n_channels):
        ax1.plot(X_within_class[:, channel, :].T, color=predefined_color_cycle[channel], label=f'channel {channel}')
    plt.xlim(0, signal_len)
    ylim = plt.ylim()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12, frameon=True)
    # plt.legend(loc='upper right', fontsize=12, frameon=True)

    if n_channels == 1:
        # plt.title("%d random test samples" % (N))
        plt.title("Misaligned signals", fontsize=title_font)
    else:
        plt.title("%d channels, %d random test samples" % (n_channels, N), fontsize=title_font)
    file_path = os.path.join(figure_path, f'misaligned_signals_class_{class_num}.svg')
    plt.savefig(file_path, format='svg')
    plt.show()

    # Misaligned Mean
    f2, ax2 = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    f2.set_size_inches(w, h)
    for channel in range(n_channels):
        ax2.plot(t, X_mean[channel], predefined_color_cycle[channel], label=f'Average signal channel {channel}')
        ax2.fill_between(t, upper[channel], lower[channel], color=predefined_color_cycle[channel], alpha=0.3,
                         label=r"$\pm\sigma$" + f" channel {channel}")
    plt.legend(loc='upper right', fontsize=12, frameon=True)
    plt.xlim(0, signal_len)
    average_ylim = plt.ylim()

    if n_channels == 1:
        plt.title("Misaligned average signal", fontsize=title_font)
    else:
        plt.title(f"{n_channels} channels, Test data mean signal ({N} samples)", fontsize=title_font)
    file_path = os.path.join(figure_path, f'misaligned_average_signals_class_{class_num}.svg')
    plt.savefig(file_path, format='svg')
    plt.show()

    # Aligned signals
    f3, ax3 = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    f3.set_size_inches(w, h)
    for channel in range(n_channels):
        ax3.plot(X_aligned_within_class[:, channel, :].T, color=predefined_color_cycle[channel],
                 label=f'channel {channel}')
    plt.title("CFDTAN aligned signals", fontsize=title_font)
    plt.xlim(0, signal_len)
    plt.ylim(ylim)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12, frameon=True)
    # plt.legend(loc='upper right', fontsize=12, frameon=True)

    file_path = os.path.join(figure_path, f'CFDTAN_aligned_signals_class_{class_num}.svg')
    plt.savefig(file_path, format='svg')
    plt.show()

    # Aligned Mean
    f4, ax4 = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    f4.set_size_inches(w, h)
    # plot transformed signal
    for channel in range(n_channels):
        ax4.plot(t, X_mean_t[channel, :], color=predefined_color_cycle[channel], label=f'Average channel {channel}')
        ax4.fill_between(t, upper_t[channel], lower_t[channel], color=predefined_color_cycle[channel], alpha=0.3,
                         label=r"$\pm\sigma$" + f" channel {channel}")

    plt.legend(loc='upper right', fontsize=12, frameon=True)
    plt.title("CFDTAN average signal", fontsize=title_font)
    plt.xlim(0, signal_len)
    plt.ylim(average_ylim)

    file_path = os.path.join(figure_path, f'CFDTAN_aligned_average_signals_class_{class_num}.svg')
    plt.savefig(file_path, format='svg')
    plt.show()


def plot_mean_signal_one_by_one(X_aligned_within_class, X_within_class, ratio, class_num, dataset_name, N=10):
    # check data dim
    import os
    figure_father_path = './figures'
    figure_path = os.path.join(figure_father_path, dataset_name)
    print(figure_path)
    os.makedirs(figure_path, exist_ok=True)

    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    # data dims: (number of samples, dim, channels)
    n_signals = len(X_within_class)  # number of samples within class

    # Sample random signals
    input_shape = X_within_class.shape[1:]  # (channels, dims) PyTorch
    signal_len = input_shape[1]
    n_channels = input_shape[0]

    indices = np.random.choice(n_signals, N)  # N samples
    X_within_class = X_within_class[indices, :, :]  # get N samples, all channels
    X_aligned_within_class = X_aligned_within_class[indices, :, :]

    # Compute mean signal and variance
    X_mean_t = np.mean(X_aligned_within_class, axis=0)
    X_std_t = np.std(X_aligned_within_class, axis=0)
    upper_t = X_mean_t + X_std_t
    lower_t = X_mean_t - X_std_t

    X_mean = np.mean(X_within_class, axis=0)
    X_std = np.std(X_within_class, axis=0)
    upper = X_mean + X_std
    lower = X_mean - X_std

    # set figure size
    [w, h] = ratio  # width, height

    title_font = 24
    # plot each channel
    for channel in range(n_channels):
        t = range(input_shape[1])
        # Misaligned Signals
        f1, ax1 = plt.subplots()
        plt.style.use('seaborn-whitegrid')
        f1.set_size_inches(w, h)
        ax1.plot(X_within_class[:, channel, :].T)
        plt.xlim(0, signal_len)

        if n_channels == 1:
            # plt.title("%d random test samples" % (N))
            plt.title("Misaligned signals", fontsize=title_font)
        else:
            plt.title("Channel: %d, %d random test samples" % (channel, N))
        file_path = os.path.join(figure_path, 'misaligned_signals.svg')
        plt.savefig(file_path, format='svg')
        plt.show()

        # Misaligned Mean
        f2, ax2 = plt.subplots()
        plt.style.use('seaborn-whitegrid')
        f2.set_size_inches(w, h)
        ax2.plot(t, X_mean[channel], 'r', label='Average signal')
        ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")
        # plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.xlim(0, signal_len)

        if n_channels == 1:
            plt.title("Misaligned average signal", fontsize=title_font)
        else:
            plt.title(f"Channel: {channel}, Test data mean signal ({N} samples)")
        file_path = os.path.join(figure_path, 'misaligned_average_signals.svg')
        plt.savefig(file_path, format='svg')
        plt.show()

        # Aligned signals
        f3, ax3 = plt.subplots()
        plt.style.use('seaborn-whitegrid')
        f3.set_size_inches(w, h)
        ax3.plot(X_aligned_within_class[:, channel, :].T)
        plt.title("CFDTAN aligned signals", fontsize=title_font)
        plt.xlim(0, signal_len)

        file_path = os.path.join(figure_path, 'CFDTAN_aligned_signals.svg')
        plt.savefig(file_path, format='svg')
        plt.show()

        # Aligned Mean
        f4, ax4 = plt.subplots()
        plt.style.use('seaborn-whitegrid')
        f4.set_size_inches(w, h)
        # plot transformed signal
        ax4.plot(t, X_mean_t[channel, :], label='Average signal')
        ax4.fill_between(t, upper_t[channel], lower_t[channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")

        # plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.title("CFDTAN average signal", fontsize=title_font)
        plt.xlim(0, signal_len)

        file_path = os.path.join(figure_path, 'CFDTAN_aligned_average_signals.svg')
        plt.savefig(file_path, format='svg')
        plt.show()


def plot_mean_signal(X_aligned_within_class, X_within_class, ratio, class_num, dataset_name, N=10):
    # check data dim
    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    # data dims: (number of samples, dim, channels)
    n_signals = len(X_within_class)  # number of samples within class

    # Sample random signals
    input_shape = X_within_class.shape[1:]  # (channels, dims) PyTorch
    signal_len = input_shape[1]
    n_channels = input_shape[0]

    indices = np.random.choice(n_signals, N)  # N samples
    X_within_class = X_within_class[indices, :, :]  # get N samples, all channels
    X_aligned_within_class = X_aligned_within_class[indices, :, :]

    # Compute mean signal and variance
    X_mean_t = np.mean(X_aligned_within_class, axis=0)
    X_std_t = np.std(X_aligned_within_class, axis=0)
    upper_t = X_mean_t + X_std_t
    lower_t = X_mean_t - X_std_t

    X_mean = np.mean(X_within_class, axis=0)
    X_std = np.std(X_within_class, axis=0)
    upper = X_mean + X_std
    lower = X_mean - X_std

    # set figure size
    [w, h] = ratio  # width, height
    f = plt.figure(1)
    # plt.style.use('seaborn-darkgrid')
    plt.style.use('seaborn-whitegrid')
    f.set_size_inches(w, n_channels * h)

    title_font = 18
    rows = n_channels * 2
    cols = 2
    plot_idx = 1
    # plot each channel
    for channel in range(n_channels):
        t = range(input_shape[1])
        # Misaligned Signals
        ax1 = f.add_subplot(rows, cols, plot_idx)
        ax1.plot(X_within_class[:, channel, :].T)
        plt.tight_layout()
        plt.xlim(0, signal_len)

        if n_channels == 1:
            # plt.title("%d random test samples" % (N))
            plt.title("Misaligned signals", fontsize=title_font)
        else:
            plt.title("Channel: %d, %d random test samples" % (channel, N))
        plot_idx += 1

        # Misaligned Mean
        ax2 = f.add_subplot(rows, cols, plot_idx)
        ax2.plot(t, X_mean[channel], 'r', label='Average signal')
        ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")
        # plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.xlim(0, signal_len)

        if n_channels == 1:
            plt.title("Misaligned average signal", fontsize=title_font)
        else:
            plt.title(f"Channel: {channel}, Test data mean signal ({N} samples)")

        plot_idx += 1

        # Aligned signals
        ax3 = f.add_subplot(rows, cols, plot_idx)
        ax3.plot(X_aligned_within_class[:, channel, :].T)
        plt.title("CFDTAN aligned signals", fontsize=title_font)
        plt.xlim(0, signal_len)

        plot_idx += 1

        # Aligned Mean
        ax4 = f.add_subplot(rows, cols, plot_idx)
        # plot transformed signal
        ax4.plot(t, X_mean_t[channel, :], label='Average signal')
        ax4.fill_between(t, upper_t[channel], lower_t[channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")

        # plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.title("CFDTAN average signal", fontsize=title_font)
        plt.xlim(0, signal_len)
        plt.tight_layout()

        plot_idx += 1

    # plt.savefig(f'{dataset_name}_{int(class_num)}.pdf', format='pdf')

    plt.suptitle(f"{dataset_name}: class-{class_num}", fontsize=title_font + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_signals(model, device, datadir, dataset_name, use_UCR=True, X_train=None, X_test=None, y_train=None,
                 y_test=None, suffix=False):
    # Close any remaining plots
    plt.close('all')

    with torch.no_grad():
        # Torch channels first
        if use_UCR:
            X_train, X_test, y_train, y_test = processed_UCR_data(datadir, dataset_name, suffix=suffix)
        else:
            assert (X_train is not None and X_test is not None and y_train is not None and y_test is not None)
            # X_train, X_test, y_train, y_test = processed_UEA_data(datadir, dataset_name)
        data = [X_train, X_test]
        labels = [y_train, y_test]
        set_names = ["train", "test"]
        for i in range(2):
            # torch dim
            X = torch.Tensor(data[i]).to(device)
            y = labels[i]
            classes = np.unique(y)
            transformed_input_tensor, thetas = model(X, return_theta=True)

            data_numpy = X.data.cpu().numpy()
            transformed_data_numpy = transformed_input_tensor.data.cpu().numpy()

            sns.set_style("whitegrid")
            # fig, axes = plt.subplots(1,2)
            for label in classes:
                class_idx = y == label
                X_within_class = data_numpy[class_idx]
                X_aligned_within_class = transformed_data_numpy[class_idx]
                # print(X_aligned_within_class.shape, X_within_class.shape)
                # 画图
                # plot_mean_signal(X_aligned_within_class, X_within_class, ratio=[10, 6],
                #                  class_num=label, dataset_name=f"{dataset_name}-{set_names[i]}")
                # plot_mean_signal_one_by_one(X_aligned_within_class, X_within_class, ratio=[10, 6],
                #                             class_num=label, dataset_name=f"{dataset_name}-{set_names[i]}")
                plot_mean_signal_multi_one_by_one(X_aligned_within_class, X_within_class, ratio=[10, 6],
                                                  class_num=label, dataset_name=f"{dataset_name}-{set_names[i]}")
