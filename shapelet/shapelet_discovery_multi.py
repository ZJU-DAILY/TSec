from typing import Union

import numpy as np
import pandas as pd
import torch

from data_helper.UEA_loader import get_UEA_data
from utils.sax_raw_data import traverse_sax
from utils.slide_raw_data import slide_ts_step_multi


def shapelet_after_sax_and_slide(ts: Union[np.ndarray, torch.Tensor], ts_labels: np.ndarray):
    slide_num = 3
    # fixme: 此处的变量的值均可进行修改
    slide_size = 0.6
    slide_step = 0.2
    w = 10
    alpha = 6

    dict_slide = {}
    dict_dim = {}
    dict_class_label = {}
    for i in range(slide_num):
        print("slide num:", i)
        ts_slide, dims, class_labels = slide_ts_step_multi(ts, alpha=slide_size, need_dim_info=True,
                                                           class_label=ts_labels)

        slide_size -= slide_step
        # 将ts_slide:(num, dim, len) 转为: (num, dim*len)
        # fixme: for the aggregation, the logic here is not the same as single variable
        # make it to (num*dim, len), which is suitable for both single and multiple
        ts_slide_con = ts_slide.reshape(ts_slide.shape[0], -1)
        # 使用sax部分对候选shapelet进行去重
        ts_pd = pd.DataFrame(ts_slide_con).loc[:, :]
        trans_ts = traverse_sax(ts_slide_con, w, alpha)
        drop_duplicate_index = trans_ts.drop_duplicates(keep='first').index
        is_unique = ts_pd.index.isin(drop_duplicate_index)
        drop_duplicate_ts = ts_slide[is_unique]
        ts_slide = drop_duplicate_ts.tolist()

        dim_pd = pd.DataFrame(dims)
        class_label_pd = pd.Series(class_labels)
        dims = dim_pd[is_unique].values
        class_labels = class_label_pd[is_unique].values

        dict_slide[str(i)] = ts_slide
        dict_dim[str(i)] = dims
        dict_class_label[str(i)] = class_labels

    max_len = int(ts.shape[2] * (slide_size + (slide_step * slide_num)))
    return dict_slide, dict_dim, dict_class_label, max_len


if __name__ == '__main__':
    data_dir = "../examples/data"
    dataset_name = "Libras"
    train_dataloader, validation_dataloader, test_dataloader, X_train, X_test, y_train, y_test = get_UEA_data(data_dir,
                                                                                                              dataset_name)
    dict_slide, dict_dim, dict_class_label = shapelet_after_sax_and_slide(X_train, y_train)
    print("end")
