import argparse
from typing import Union

import numpy as np
import pandas as pd
import torch

from utils.sax_raw_data import traverse_sax
from utils.slide_raw_data import slide_ts_step, slide_ts_step_single, slide_ts_step_aggr


def shapelet_after_sax_and_slide(ts: Union[np.ndarray, torch.Tensor], ts_labels: np.ndarray):
    slide_num = 3
    # fixme: 此处的变量的值均可进行修改
    slide_size = 0.6
    slide_step = 0.2
    w = 10
    alpha = 7

    dict_slide = {}
    dict_dim = {}
    dict_class_label = {}
    for i in range(slide_num):
        ts_slide, dims, class_labels, indexes = slide_ts_step(ts, alpha=slide_size, need_dim_info=True,
                                                              class_label=ts_labels)

        slide_size -= slide_step

        # 使用sax部分对候选shapelet进行去重
        ts_pd = pd.DataFrame(ts_slide).loc[:, :]
        trans_ts = traverse_sax(ts_slide, w, alpha)
        drop_duplicate_index = trans_ts.drop_duplicates(keep='first').index
        is_unique = ts_pd.index.isin(drop_duplicate_index)
        drop_duplicate_ts = ts_pd[is_unique]
        ts_slide = drop_duplicate_ts.values

        dim_pd = pd.Series(dims)
        class_label_pd = pd.Series(class_labels)
        dims = dim_pd[is_unique].values
        class_labels = class_label_pd[is_unique].values

        dict_slide[str(i)] = ts_slide
        dict_dim[str(i)] = dims
        dict_class_label[str(i)] = class_labels

    return dict_slide, dict_dim, dict_class_label


def shapelet_return_sax_and_slide(ts: Union[np.ndarray, torch.Tensor], ts_labels: np.ndarray):
    slide_num = 2
    # fixme: 此处的变量的值均可进行修改
    slide_size = 0.6
    slide_step = 0.3
    w = 6
    alpha = 4

    dict_slide = {}
    dict_dim = {}
    dict_class_label = {}
    dict_sax = {}
    dict_index = {}
    for i in range(slide_num):
        ts_slide, dims, class_labels, indexes = slide_ts_step_single(ts, alpha=slide_size, need_dim_info=True,
                                                                     class_label=ts_labels)

        slide_size -= slide_step

        # # 使用sax部分对候选shapelet进行去重
        # ts_pd = pd.DataFrame(ts_slide).loc[:, :]
        # trans_ts = traverse_sax(ts_slide, w, alpha)
        # drop_duplicate_index = trans_ts.drop_duplicates(keep='first').index
        # is_unique = ts_pd.index.isin(drop_duplicate_index)
        # drop_duplicate_ts = ts_pd[is_unique]
        # ts_slide = drop_duplicate_ts.values
        # ts_sax = trans_ts[is_unique].values
        # ts_index = indexes[is_unique]

        # dim_pd = pd.Series(dims)
        # class_label_pd = pd.Series(class_labels)
        # dims = dim_pd[is_unique].values
        # class_labels = class_label_pd[is_unique].values

        dict_slide[str(i)] = np.array(ts_slide)
        dict_dim[str(i)] = dims
        dict_class_label[str(i)] = np.array(class_labels)

    return dict_slide, dict_dim, dict_class_label, dict_sax, dict_index


def shapelet_aggr_sax_and_slide(ts: Union[np.ndarray, torch.Tensor], ts_labels: np.ndarray,
                                parser: argparse.Namespace, is_multi=False):
    # slide_num = 2
    slide_size = parser.ratio
    slide_step = parser.slide_step
    # w = 6
    # alpha = 4

    dict_slide = {}
    dict_dim = {}
    dict_class_label = {}

    for i in range(parser.slide_time):
        ts_slide, dims, class_labels = slide_ts_step_aggr(ts, ratio=slide_size, w=parser.w, alpha=parser.alpha,
                                                          extra_info=True, class_label=ts_labels)
        slide_size -= slide_step

        dict_slide[str(i)] = np.array(ts_slide)
        dict_dim[str(i)] = np.array(dims)
        dict_class_label[str(i)] = np.array(class_labels, dtype=int)

    if not is_multi:
        return dict_slide, dict_dim, dict_class_label
    else:
        return dict_slide, dict_dim, dict_class_label, int(ts.shape[2] * parser.ratio), ts.shape[1]
