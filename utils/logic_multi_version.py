import argparse
import multiprocessing
from typing import List, Dict, Union

import numpy as np
import torch
import torch.multiprocessing as mp

max_num_cores = int(multiprocessing.cpu_count() / 4)


def get_cf_dataloader(data: Dict[str, np.ndarray], batch_size):
    from data_helper.UCR_loader import np_to_dataloader, get_train_and_validation_loaders

    train_dataloader = np_to_dataloader(data['X'], data['y'], batch_size=batch_size, shuffle=True)
    train_dataloader, validation_dataloader = get_train_and_validation_loaders(train_dataloader, validation_split=0.1,
                                                                               batch_size=batch_size)
    return train_dataloader, validation_dataloader


def shapelet_aggr_sax_and_slide_parallely(ts: Union[np.ndarray, torch.Tensor], ts_labels: np.ndarray,
                                          parser: argparse.Namespace, is_multi=False):
    from utils.slide_raw_data import slide_ts_step_aggr
    # slide_num = 2
    slide_size = parser.ratio
    slide_step = parser.slide_step
    # w = 6
    # alpha = 4

    dict_slide = {}
    dict_dim = {}
    dict_class_label = {}

    async_result_list = []
    num_cores = parser.slide_time if parser.slide_time < max_num_cores else max_num_cores
    pool = multiprocessing.Pool(processes=num_cores)
    for i in range(parser.slide_time):
        async_result = pool.apply_async(func=slide_ts_step_aggr, args=(ts,),
                                        kwds={'ratio': slide_size - i * slide_step, 'w': parser.w,
                                              'alpha': parser.alpha, 'extra_info': True, 'class_label': ts_labels, })
        async_result_list.append(async_result)
    for idx, result in enumerate(async_result_list):
        ts_slide, dims, class_labels = result.get()

        # ts_slide, dims, class_labels = slide_ts_step_aggr(ts, ratio=slide_size, w=parser.w, alpha=parser.alpha,
        #                                                   extra_info=True, class_label=ts_labels)
        # slide_size -= slide_step

        dict_slide[str(idx)] = np.array(ts_slide)
        dict_dim[str(idx)] = np.array(dims)
        dict_class_label[str(idx)] = np.array(class_labels, dtype=int)
    pool.close()
    pool.join()
    if not is_multi:
        return dict_slide, dict_dim, dict_class_label
    else:
        return dict_slide, dict_dim, dict_class_label, int(ts.shape[2] * parser.ratio), ts.shape[1]


def process_single_CFDTAN(data: Dict[str, np.ndarray], args: argparse.Namespace, dataset_name: str, idx: int,
                          device: torch.device):
    from utils.train_model import train_T
    temp_single_train_loader, temp_single_validation_loader = get_cf_dataloader(data, args.cf_batch_size)
    temp_single_model = train_T(temp_single_train_loader, temp_single_validation_loader, args, print_model=False,
                                suffix=dataset_name + '_alignment_single_' + str(idx))
    if args.plot_cf_results:
        # todo: also here pass
        pass
    temp_single_X = torch.Tensor(data['X']).to(device)
    trans_temp_single_tensor = temp_single_model(temp_single_X, return_theta=False)
    trans_temp_single_X = trans_temp_single_tensor.cpu().detach().numpy()
    data['X'] = trans_temp_single_X
    return idx


def process_single_uni_classification(data: Dict[str, Dict[str, np.ndarray]], args: argparse.Namespace, idx: int,
                                      dataset_name: str):
    from classification.uni_lstm import train_uni_classifier_sim
    total_temp_single_X_list = []
    total_temp_single_y_list = []

    for key in data['X'].keys():
        total_temp_single_X_list.extend(data['X'][key].tolist())
    for key in data['y'].keys():
        total_temp_single_y_list.extend(data['y'][key].astype('int').tolist())

    # unique_values, counts = np.unique(total_temp_single_y_list, return_counts=True)
    # print("train unique_values:{},counts:{}".format(unique_values, counts))

    num_classes = len(set(total_temp_single_y_list))
    temp_single_model = train_uni_classifier_sim(data=total_temp_single_X_list, y=total_temp_single_y_list,
                                                 num_class=num_classes, args=args)
    # torch.save(temp_single_model.state_dict(),
    #            f'./checkpoints/{dataset_name}_classification_single_{idx}_checkpoint.pth')
    torch.save(temp_single_model,
               f'./checkpoints/classification/single/{dataset_name}_classification_single_{idx}_checkpoint.pth')
    # return temp_single_model


def process_CFDTAN_parallely(
        args: argparse.Namespace, dataset_name: str,
        list_single_part: List[Dict[str, np.ndarray]],
        multi_part: Dict[str, np.ndarray] = None
):
    from utils.train_model import train_T
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async_result_list = []
    single_model_list = []
    # single part
    # multiprocessing.set_start_method('spawn')
    num_processes = len(list_single_part) if len(list_single_part) != 0 else 1
    num_processes = num_processes if num_processes < max_num_cores else max_num_cores
    # single_pool = multiprocessing.Pool(processes=num_processes)
    single_pool = mp.get_context('spawn').Pool(processes=num_processes)
    for idx in range(len(list_single_part)):
        async_result = single_pool.apply_async(func=process_single_CFDTAN,
                                               args=(list_single_part[idx], args, dataset_name, idx, device,),
                                               callback=async_result_list.append)
    for result in async_result_list:
        result.get()

    single_pool.close()
    single_pool.join()

    if multi_part is not None:
        # multi-part
        multi_train_loader, multi_validation_loader = get_cf_dataloader(multi_part, args.cf_batch_size)
        multi_model = train_T(multi_train_loader, multi_validation_loader, args, print_model=False,
                              suffix=dataset_name + '_alignment_multi', is_multi=True)
        if args.plot_cf_results:
            # todo: the logic of plotting need modifying, so here only pass
            pass
        # here I do the return in the way of modifying the parameter in dictionary format
        multi_X = torch.Tensor(multi_part['X']).to(device)
        trans_multi_tensor = multi_model(multi_X, return_theta=False)
        trans_multi_X = trans_multi_tensor.cpu().detach().numpy()
        multi_part['X'] = trans_multi_X

    #     return multi_model, single_model_list
    # return single_model_list


def process_sax_slide_parallely(
        args: argparse.Namespace,
        list_single_part: List[Dict[str, np.ndarray]],
        multi_part: Dict[str, np.ndarray] = None
):
    from shapelet.shapelet_discovery import shapelet_aggr_sax_and_slide

    if multi_part is not None:
        # multi-part
        dict_multi_slide, dict_multi_dim, dict_multi_class_label, \
            max_len, num_channels = shapelet_aggr_sax_and_slide(multi_part['X'],
                                                                multi_part['y'],
                                                                parser=args,
                                                                is_multi=True)
        # also here, just modifying the parameter passing in
        multi_part['X'] = dict_multi_slide
        multi_part['y'] = dict_multi_class_label
        multi_part['dim'] = dict_multi_dim
        multi_part['max_len'] = max_len
        multi_part['num_channels'] = num_channels

    # single part
    async_result_list = []
    num_processes = len(list_single_part) if len(list_single_part) != 0 else 1
    num_processes = num_processes if num_processes < max_num_cores else max_num_cores
    single_pool = multiprocessing.Pool(processes=num_processes)
    for idx in range(len(list_single_part)):
        async_result = single_pool.apply_async(func=shapelet_aggr_sax_and_slide,
                                               args=(list_single_part[idx]['X'], list_single_part[idx]['y'],),
                                               kwds={'parser': args, })
        async_result_list.append(async_result)
    for idx, result in enumerate(async_result_list):
        temp_dict_single_slide, temp_dict_single_dim, temp_dict_single_class_label = result.get()

        # temp_dict_single_slide, temp_dict_single_dim, temp_dict_single_class_label = shapelet_aggr_sax_and_slide(
        #     list_single_part[idx]['X'], list_single_part[idx]['y'], parser=args)
        list_single_part[idx]['X'] = temp_dict_single_slide
        list_single_part[idx]['y'] = temp_dict_single_class_label
        list_single_part[idx]['dim'] = temp_dict_single_dim
    single_pool.close()
    single_pool.join()


def process_classification_parallely(
        args: argparse.Namespace,
        list_single_part: List[Dict[str, Dict[str, np.ndarray]]],
        dataset_name: str,
        multi_part: Dict[str, Union[Dict[str, np.ndarray], int]] = None
):
    from classification.multi_gcn import train_multi_classifier_sim

    single_model_list = []
    async_result_list = []
    # multiprocessing.set_start_method('spawn')
    num_processes = len(list_single_part) if len(list_single_part) != 0 else 1
    num_processes = num_processes if num_processes < max_num_cores else max_num_cores
    # single_pool = multiprocessing.Pool(processes=num_processes)
    single_pool = mp.get_context('spawn').Pool(processes=num_processes)
    # single part
    for idx in range(len(list_single_part)):
        single_pool.apply_async(func=process_single_uni_classification,
                                args=(list_single_part[idx], args, idx, dataset_name,))
        # async_result_list.append(async_result)
    # for result in async_result_list:
    #     temp_single_model = result.get()
    # total_temp_single_X_list = []
    # total_temp_single_y_list = []
    #
    # for key in list_single_part[idx]['X'].keys():
    #     total_temp_single_X_list.extend(list_single_part[idx]['X'][key].tolist())
    # for key in list_single_part[idx]['y'].keys():
    #     total_temp_single_y_list.extend(list_single_part[idx]['y'][key].astype('int').tolist())
    #
    # # unique_values, counts = np.unique(total_temp_single_y_list, return_counts=True)
    # # print("train unique_values:{},counts:{}".format(unique_values, counts))
    #
    # num_classes = len(set(total_temp_single_y_list))
    # temp_single_model = train_uni_classifier_sim(data=total_temp_single_X_list, y=total_temp_single_y_list,
    #                                              num_class=num_classes, args=args)
    # single_model_list.append(temp_single_model)
    single_pool.close()
    single_pool.join()

    if multi_part is not None:
        # multi part
        total_multi_X_list = []
        total_multi_y_list = []
        for key in multi_part['X'].keys():
            total_multi_X_list.extend(multi_part['X'][key].tolist())
        for key in multi_part['y'].keys():
            total_multi_y_list.extend(multi_part['y'][key].astype('int').tolist())

        # unique_values, counts = np.unique(total_multi_y_list, return_counts=True)
        # print("train unique_values:{},counts:{}".format(unique_values, counts))

        num_classes = len(set(total_multi_y_list))
        multi_model = train_multi_classifier_sim(data=total_multi_X_list, y=total_multi_y_list, num_class=num_classes,
                                                 max_len=multi_part['max_len'], num_channels=multi_part['num_channels'],
                                                 args=args)
        # torch.save(multi_model.state_dict(),
        #            f'./checkpoints/{dataset_name}_classification_multi_checkpoint.pth')
        torch.save(multi_model,
                   f'./checkpoints/classification/multiple/{dataset_name}_classification_multi_checkpoint.pth')
    #     return multi_model, single_model_list
    # return single_model_list
