import math
from typing import Union

import numpy as np
import pandas as pd
import torch

from utils.sax_raw_data import traverse_sax


def slide_ts_step(ts: Union[np.ndarray, torch.Tensor], alpha, need_dim_info=False, class_label=None):
    assert (len(ts.shape) == 3)
    num_old, dim_old, len_old = ts.shape
    len_new = int(len_old * alpha)

    ts_alpha = ts[:, :, 0: len_new]

    if len_old <= 50:
        step = 1
    elif len_old <= 100:
        step = 2
    elif len_old <= 300:
        step = 3
    elif len_old <= 1000:
        step = 4
    elif len_old <= 1500:
        step = 5
    elif len_old <= 2000:
        step = 7
    elif len_old <= 3000:
        step = 10
    else:
        step = 100

    step_num = int(math.ceil((len_old - len_new) / step))

    for k in range(1, len_old - len_new + 1, step):
        ts_temp = ts[:, :, k: len_new + k]
        if isinstance(ts_temp, np.ndarray):
            ts_alpha = np.concatenate((ts_alpha, ts_temp), axis=0)
        elif isinstance(ts_temp, torch.Tensor):
            ts_alpha = torch.cat((ts_alpha, ts_temp), dim=0)
        else:
            raise Exception("Unknown data type!")

    ts_beta = ts_alpha.reshape(num_old * dim_old * (step_num + 1), len_new)

    if not need_dim_info:
        return ts_beta
    else:
        assert (class_label is not None)
        class_label = torch.Tensor(class_label)
        dims_info = [idx for idx in range(dim_old)]
        class_labels = class_label.repeat(dim_old * (step_num + 1))
        dims = dims_info * (num_old * (step_num + 1))
        indexes = np.repeat(range(num_old), (step_num + 1))
        return ts_beta, dims, class_labels, indexes


def slide_ts_step_single(ts: Union[np.ndarray, torch.Tensor], alpha, need_dim_info=False, class_label=None):
    assert (len(ts.shape) == 3)
    num_old, dim_old, len_old = ts.shape
    len_new = int(len_old * alpha)

    if len_old <= 50:
        step = 1
    elif len_old <= 100:
        step = 2
    elif len_old <= 300:
        step = 3
    elif len_old <= 1000:
        step = 4
    elif len_old <= 1500:
        step = 5
    elif len_old <= 2000:
        step = 7
    elif len_old <= 3000:
        step = 10
    else:
        step = 100

    step_num = int(math.ceil((len_old - len_new) / step))
    ts_single = []
    class_labels = []
    for i in range(num_old):
        ts_alpha = ts[i, :, 0: len_new]
        for k in range(1, len_old - len_new + 1, step):
            ts_temp = ts[i, :, k: len_new + k]
            if isinstance(ts_temp, np.ndarray):
                ts_alpha = np.concatenate((ts_alpha, ts_temp), axis=0)
            elif isinstance(ts_temp, torch.Tensor):
                ts_alpha = torch.cat((ts_alpha, ts_temp), dim=0)
            else:
                raise Exception("Unknown data type!")
        ts_pd = pd.DataFrame(ts_alpha).loc[:, :]
        trans_ts = traverse_sax(ts_alpha, 6, 4)
        drop_duplicate_index = trans_ts.drop_duplicates(keep='first').index
        is_unique = ts_pd.index.isin(drop_duplicate_index)
        drop_duplicate_ts = ts_pd[is_unique]
        ts_alpha = drop_duplicate_ts.values

        ts_single.extend(np.array(ts_alpha))
        class_labels.extend(np.repeat(class_label[i], len(ts_alpha)))

    return ts_single, None, class_labels, None


def slide_ts_step_multi(ts: Union[np.ndarray, torch.Tensor], alpha, need_dim_info=False, class_label=None):
    assert (len(ts.shape) == 3)
    num_old, dim_old, len_old = ts.shape
    len_new = int(len_old * alpha)

    ts_alpha = ts[:, :, 0: len_new]

    if len_old <= 50:
        step = 1
    elif len_old <= 100:
        step = 2
    elif len_old <= 300:
        step = 3
    elif len_old <= 1000:
        step = 4
    elif len_old <= 1500:
        step = 5
    elif len_old <= 2000:
        step = 7
    elif len_old <= 3000:
        step = 10
    else:
        step = 100

    step_num = int(math.ceil((len_old - len_new) / step))

    for k in range(1, len_old - len_new + 1, step):
        ts_temp = ts[:, :, k: len_new + k]
        if isinstance(ts_temp, np.ndarray):
            ts_alpha = np.concatenate((ts_alpha, ts_temp), axis=0)
        elif isinstance(ts_temp, torch.Tensor):
            ts_alpha = torch.cat((ts_alpha, ts_temp), dim=0)
        else:
            raise Exception("Unknown data type!")

    if not need_dim_info:
        return ts_alpha
    else:
        assert (class_label is not None)
        dims_info = [idx for idx in range(dim_old)]
        class_labels = class_label.repeat(step_num + 1)
        # dims = dims_info * (num_old * (step_num + 1))
        dims = np.tile(dims_info, (num_old * (step_num + 1), 1))

        return ts_alpha, dims, class_labels


def slide_ts_step_aggr(ts: Union[np.ndarray, torch.Tensor], ratio, w, alpha, extra_info=False, class_label=None):
    assert (len(ts.shape) == 3)
    num_old, dim_old, len_old = ts.shape
    len_new = int(len_old * ratio)

    if len_old <= 50:
        step = 1
    elif len_old <= 100:
        step = 2
    elif len_old <= 300:
        step = 3
    elif len_old <= 1000:
        step = 4
    elif len_old <= 1500:
        step = 5
    elif len_old <= 2000:
        step = 7
    elif len_old <= 3000:
        step = 10
    else:
        step = 100

    step_num = int(math.ceil((len_old - len_new) / step))
    ts_aggr = []
    class_labels = []
    dims = []
    if extra_info:
        assert (class_label is not None)

    for i in range(num_old):
        ts_alpha = ts[i, :, 0: len_new]
        for k in range(1, len_old - len_new + 1, step):
            ts_temp = ts[i, :, k: len_new + k]
            if isinstance(ts_temp, np.ndarray):
                ts_alpha = np.concatenate((ts_alpha, ts_temp), axis=0)
            elif isinstance(ts_temp, torch.Tensor):
                ts_alpha = torch.cat((ts_alpha, ts_temp), dim=0)
            else:
                raise Exception("Unknown data type!")

        remain_list = list(range(0, step_num + 1))
        for dim in range(dim_old):
            element_idx = list(range(dim, ts_alpha.shape[0], dim_old))
            single_channel_data = ts_alpha[element_idx]
            single_trans_ts = traverse_sax(single_channel_data, w, alpha)
            drop_duplicate_index = single_trans_ts.drop_duplicates(keep='first').index.tolist()
            remain_list = list(set(remain_list) & set(drop_duplicate_index))

        idx_list = []
        for idx in remain_list:
            idx_list.extend([idx * dim_old + num for num in range(dim_old)])

        ts_alpha = ts_alpha[idx_list]
        dims_info = [idx for idx in range(dim_old)]
        temp_dims = dims_info * len(remain_list)
        assert (len(temp_dims) == ts_alpha.shape[0])

        # ts_pd = pd.DataFrame(ts_alpha).loc[:, :]
        # trans_ts = traverse_sax(ts_alpha, w, alpha)
        # drop_duplicate_index = trans_ts.drop_duplicates(keep='first').index
        # is_unique = ts_pd.index.isin(drop_duplicate_index)
        # drop_duplicate_ts = ts_pd[is_unique]
        # ts_alpha = drop_duplicate_ts.values
        #
        # dim_pd = pd.Series(temp_dims)
        # temp_dims = dim_pd[is_unique].values

        ts_aggr.extend(np.array(ts_alpha))
        dims.extend(np.array(temp_dims))
        if extra_info:
            class_labels.extend(np.repeat(class_label[i], ts_alpha.shape[0]))

    # here, whether the original data is numpy.ndarray or torch.Tensor
    # the return value is always list of numpy.ndarray
    if extra_info:
        return ts_aggr, dims, class_labels
    else:
        return ts_aggr
