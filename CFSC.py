import argparse
from typing import Tuple, List, Dict, Union

import numpy as np
import torch
import time
import json

from CFDTAN.CFDTAN_layer import CfDtAn
from classification.multi_gcn import GCLSTM as MultiCla
from classification.uni_lstm import BiLSTMAttentionClassifier as SingleCla
from utils.logic_multi_version import process_CFDTAN_parallely, process_sax_slide_parallely, \
    process_classification_parallely


# todo: here the following args do not include the network layer parameters yet
def CFDTAN_arg_parser():
    parser = argparse.ArgumentParser(description='CFDTAN args')
    parser.add_argument('--tess_size', type=int, default=16,
                        help="CPA velocity field partition")
    parser.add_argument('--smoothness_prior', default=True,
                        help="smoothness prior flag", action='store_true')
    parser.add_argument('--no_smoothness_prior', dest='smoothness_prior', default=True,
                        help="no smoothness prior flag", action='store_false')
    parser.add_argument('--lambda_smooth', type=float, default=1,
                        help="lambda_smooth, larger values -> smoother warps")
    parser.add_argument('--lambda_var', type=float, default=0.1,
                        help="lambda_var, larger values -> larger warps")
    parser.add_argument('--n_recurrences', type=int, default=1,
                        help="number of recurrences of R-CfDtAn")
    parser.add_argument('--zero_boundary', type=bool, default=True,
                        help="zero boundary constrain")
    parser.add_argument('--cf_n_epochs', type=int, default=100,
                        help="CFDTAN phase number of epochs")
    parser.add_argument('--cf_batch_size', type=int, default=64,
                        help="CFDTAN phase batch size")
    parser.add_argument('--cf_lr', type=float, default=0.0001,
                        help="CFDTAN phase learning rate")
    parser.add_argument('--n_ss', type=int, default=8, help="times for scaling and squaring")
    parser.add_argument('--back_version', type=bool, default=False, help="back to the original network version")
    parser.add_argument('--plot_cf_results', type=bool, default=False, help="plot the time series after CFDTAN net")
    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def sax_slide_arg_parser(unknown_args: List[str]):
    parser = argparse.ArgumentParser(description='sax and slide args')
    parser.add_argument('--ratio', type=float, default=0.6, help="initial slide ratio")
    parser.add_argument('--slide_step', type=float, default=0.3, help="des-step slide ratio")
    parser.add_argument('--slide_time', type=int, default=2, help="slide times")
    parser.add_argument('--w', type=int, default=6, help="length to cal the average in sax")
    parser.add_argument('--alpha', type=int, default=4, help="the size of vocabulary in sax")
    # args = parser.parse_args()
    args, unknown_args_ = parser.parse_known_args(unknown_args)
    return args, unknown_args_


def classification_arg_parser(unknown_args: List[str]):
    parser = argparse.ArgumentParser(description='classification nets args')
    parser.add_argument('--cla_batch_size', type=int, default=128, help="classification phase batch size")
    parser.add_argument('--cla_n_epochs', type=int, default=200,
                        help="classification phase number of epochs")
    parser.add_argument('--cla_lr', type=float, default=0.001,
                        help="classification phase learning rate")
    parser.add_argument('--uni_embedding_dim', type=int, default=320, help="embedding dim for Time2Vec")
    parser.add_argument('--uni_hidden_dim', type=int, default=128, help="hidden dim for bi-lstm")
    parser.add_argument('--multi_hidden_dim', type=int, default=20, help="hidden dim for the GCN")
    parser.add_argument('--multi_num_layers', type=int, default=2, help="num layers of the lstm in the multi net")
    # args = parser.parse_args()
    args, unknown_args_ = parser.parse_known_args(unknown_args)
    return args, unknown_args_


def file_arg_parser(unknown_args: List[str]):
    parser = argparse.ArgumentParser(description='args for file store path')
    parser.add_argument('--folder_path', type=str, default="./examples/data/arff/Multivariate_arff",
                        help="the dataset folder path")
    parser.add_argument('--dataset_name', type=str, default="EigenWorms", help="the dataset name")
    parser.add_argument('--is_multi', type=bool, default=True, help="whether the dataset is multi-variable")
    parser.add_argument('--need_path_suffix', type=bool, default=False, help="whether the file has a '.' suffix or not")
    parser.add_argument('--need_test', type=bool, default=False, help="the whole workflow need test part or not")
    parser.add_argument('--choice_ratio', type=float, default=0.3, help="used in sample multi data")
    parser.add_argument('--multi_process_acc', type=bool, default=True,
                        help="whether to use multi_process acceleration")
    # args = parser.parse_args()
    args, unknown_args_ = parser.parse_known_args(unknown_args)
    return args, unknown_args_


def get_cf_dataloader(data: Dict[str, np.ndarray], batch_size):
    from data_helper.UCR_loader import np_to_dataloader, get_train_and_validation_loaders

    train_dataloader = np_to_dataloader(data['X'], data['y'], batch_size=batch_size, shuffle=True)
    train_dataloader, validation_dataloader = get_train_and_validation_loaders(train_dataloader, validation_split=0.1,
                                                                               batch_size=batch_size)
    return train_dataloader, validation_dataloader


# the former part returns the multiple part ts, the latter part returns the list of single part ts
def cal_correlation(args: argparse.Namespace) \
        -> Union[List[Dict[str, np.ndarray]], Tuple[
            Dict[str, np.ndarray], List[Dict[str, np.ndarray]], List[int], List[int]]]:
    from data_helper.UEA_loader import processed_UEA_data
    from data_helper.UCR_loader import processed_UCR_data
    from utils.complete_subgraph import get_largest_complete_subgraph

    wrapper_list_single_data = []
    # here just abandon the test data for this progress is training workflow
    # in test, we can just reload the data again and just get the test data
    if args.is_multi:
        X_train, _, y_train, _ = processed_UEA_data(path=args.folder_path, dataset=args.dataset_name)
        assert (len(X_train.shape) == 3)
        data_num, var_num, _ = X_train.shape
        choice_num = int(data_num * args.choice_ratio)
        divisor_mat = np.ones((var_num, var_num)) * choice_num
        dividend_mat = np.zeros((var_num, var_num))

        sampled_indices = np.random.choice(data_num, size=choice_num, replace=False)
        choice_X_train = X_train[sampled_indices]

        for i in range(choice_num):
            temp_data = choice_X_train[i]
            temp_corr_mat = np.abs(np.corrcoef(temp_data))
            temp_rela_indices = temp_corr_mat >= 0.7
            dividend_mat[temp_rela_indices] += 1

        # dividend_mat = np.triu(dividend_mat, k=1)
        result_mat = dividend_mat / divisor_mat
        result_indices = result_mat > 0.5
        adj_matrix = np.zeros((var_num, var_num))
        adj_matrix[result_indices] = 1
        np.fill_diagonal(adj_matrix, 0)

        # mul_var = find_max_connected_subgraph(adj_matrix)
        mul_var = get_largest_complete_subgraph(adj_matrix)
        # add logic for no relevant variables
        wrapper_dict_multi_data = None
        if len(mul_var) != 1:
            single_var = list(set(range(var_num)).difference(mul_var))
            mul_var = list(mul_var)

            wrapper_dict_multi_data = {'X': X_train[:, mul_var, :], 'y': y_train}
            # assert (len(single_var) != 0)
        else:
            mul_var = []
            single_var = list(range(var_num))

        for idx in single_var:
            wrapper_dict_single_data = {'X': np.expand_dims(X_train[:, idx, :], 1), 'y': y_train}
            wrapper_list_single_data.append(wrapper_dict_single_data)
        return wrapper_dict_multi_data, wrapper_list_single_data, mul_var, single_var
    else:
        X_train, _, y_train, _ = processed_UCR_data(datadir=args.folder_path, dataset=args.dataset_name,
                                                    suffix=args.need_path_suffix)
        wrapper_dict_single_data = {'X': X_train, 'y': y_train}
        wrapper_list_single_data.append(wrapper_dict_single_data)
        return wrapper_list_single_data


def process_CFDTAN(
        args: argparse.Namespace, dataset_name: str,
        list_single_part: List[Dict[str, np.ndarray]],
        multi_part: Dict[str, np.ndarray] = None
) -> Union[List[CfDtAn], Tuple[CfDtAn, List[CfDtAn]]]:
    from utils.train_model import train_T
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    single_model_list = []
    # single part
    for idx in range(len(list_single_part)):
        temp_single_train_loader, temp_single_validation_loader = get_cf_dataloader(list_single_part[idx],
                                                                                    args.cf_batch_size)
        temp_single_model = train_T(temp_single_train_loader, temp_single_validation_loader, args, print_model=False,
                                    suffix=dataset_name + '_single_' + str(idx))
        if args.plot_cf_results:
            # todo: also here pass
            pass
        temp_single_X = torch.Tensor(list_single_part[idx]['X']).to(device)
        trans_temp_single_tensor = temp_single_model(temp_single_X, return_theta=False)
        trans_temp_single_X = trans_temp_single_tensor.cpu().detach().numpy()
        list_single_part[idx]['X'] = trans_temp_single_X
        single_model_list.append(temp_single_model)

    if multi_part is not None:
        # multi-part
        multi_train_loader, multi_validation_loader = get_cf_dataloader(multi_part, args.cf_batch_size)
        multi_model = train_T(multi_train_loader, multi_validation_loader, args, print_model=False,
                              suffix=dataset_name + '_multi', is_multi=True)
        if args.plot_cf_results:
            # todo: the logic of plotting need modifying, so here only pass
            pass
        # here I do the return in the way of modifying the parameter in dictionary format
        multi_X = torch.Tensor(multi_part['X']).to(device)
        trans_multi_tensor = multi_model(multi_X, return_theta=False)
        trans_multi_X = trans_multi_tensor.cpu().detach().numpy()
        multi_part['X'] = trans_multi_X

        return multi_model, single_model_list
    return single_model_list


def process_sax_slide(
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
    for idx in range(len(list_single_part)):
        temp_dict_single_slide, temp_dict_single_dim, temp_dict_single_class_label = shapelet_aggr_sax_and_slide(
            list_single_part[idx]['X'], list_single_part[idx]['y'], parser=args)
        list_single_part[idx]['X'] = temp_dict_single_slide
        list_single_part[idx]['y'] = temp_dict_single_class_label
        list_single_part[idx]['dim'] = temp_dict_single_dim


def process_classification(
        args: argparse.Namespace,
        list_single_part: List[Dict[str, Dict[str, np.ndarray]]],
        multi_part: Dict[str, Union[Dict[str, np.ndarray], int]] = None
) -> Union[List[SingleCla], Tuple[MultiCla, List[SingleCla]]]:
    from classification.multi_gcn import train_multi_classifier_sim
    from classification.uni_lstm import train_uni_classifier_sim

    single_model_list = []
    # single part
    for idx in range(len(list_single_part)):
        total_temp_single_X_list = []
        total_temp_single_y_list = []

        for key in list_single_part[idx]['X'].keys():
            total_temp_single_X_list.extend(list_single_part[idx]['X'][key].tolist())
        for key in list_single_part[idx]['y'].keys():
            total_temp_single_y_list.extend(list_single_part[idx]['y'][key].astype('int').tolist())

        # unique_values, counts = np.unique(total_temp_single_y_list, return_counts=True)
        # print("train unique_values:{},counts:{}".format(unique_values, counts))

        num_classes = len(set(total_temp_single_y_list))
        temp_single_model = train_uni_classifier_sim(data=total_temp_single_X_list, y=total_temp_single_y_list,
                                                     num_class=num_classes, args=args)
        single_model_list.append(temp_single_model)

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

        return multi_model, single_model_list
    return single_model_list


if __name__ == "__main__":
    cf_args, unknown_arg = CFDTAN_arg_parser()
    file_args, unknown_arg1 = file_arg_parser(unknown_arg)
    # print(file_args)
    # print(unknown_arg)
    sax_slide_args, unknown_arg2 = sax_slide_arg_parser(unknown_arg1)
    cla_args, unknown_arg3 = classification_arg_parser(unknown_arg2)
    assert (len(unknown_arg3) == 0)

    if file_args.multi_process_acc:
        if file_args.is_multi:
            cor_start_time = time.time()
            multi_data, list_single_data, multi_index, single_index = cal_correlation(file_args)
            cor_end_time = time.time()
            index_dict = {'single_index': single_index, 'multi_index': multi_index}
            with open(f'./channel_index/{file_args.dataset_name}.json', 'w') as f:
                json.dump(index_dict, f)

            if len(single_index) > 50:
                raise Exception("too many single index!")
        else:
            cor_start_time = time.time()
            list_single_data = cal_correlation(file_args)
            cor_end_time = time.time()
            multi_data = None

        ali_start_time = time.time()
        process_CFDTAN_parallely(args=cf_args,
                                 dataset_name=file_args.dataset_name,
                                 list_single_part=list_single_data,
                                 multi_part=multi_data)
        ali_end_time = time.time()

        sax_start_time = time.time()
        process_sax_slide_parallely(args=sax_slide_args, list_single_part=list_single_data,
                                    multi_part=multi_data)
        sax_end_time = time.time()

        cla_start_time = time.time()
        process_classification_parallely(args=cla_args,
                                         list_single_part=list_single_data,
                                         multi_part=multi_data, dataset_name=file_args.dataset_name)
        cla_end_time = time.time()
        # else:
        #     list_single_data = cal_correlation(file_args)
        #     process_CFDTAN_parallely(args=cf_args, dataset_name=file_args.dataset_name,
        #                              list_single_part=list_single_data)
        #     process_sax_slide_parallely(args=sax_slide_args, list_single_part=list_single_data)
        #     process_classification_parallely(args=cla_args, list_single_part=list_single_data,
        #                                      dataset_name=file_args.dataset_name)

        record_time = {'cor_time': round(cor_end_time - cor_start_time, 3),
                       'ali_time': round(ali_end_time - ali_start_time, 3),
                       'sax_time': round(sax_end_time - sax_start_time, 3),
                       'cla_time': round(cla_end_time - cla_start_time, 3)}
        with open(f'./record_time/{file_args.dataset_name}.json', 'w') as f:
            json.dump(record_time, f)
    else:
        if file_args.is_multi:
            multi_data, list_single_data = cal_correlation(file_args)
            if multi_data is None:
                cf_list_single_model = process_CFDTAN(args=cf_args, dataset_name=file_args.dataset_name,
                                                      list_single_part=list_single_data)
                process_sax_slide(args=sax_slide_args, list_single_part=list_single_data)
                cla_list_single_model = process_classification(args=cla_args, list_single_part=list_single_data)
            else:
                cf_multi_model, cf_list_single_model = process_CFDTAN(args=cf_args, dataset_name=file_args.dataset_name,
                                                                      list_single_part=list_single_data,
                                                                      multi_part=multi_data)
                process_sax_slide(args=sax_slide_args, list_single_part=list_single_data, multi_part=multi_data)
                cla_multi_model, cla_list_single_model = process_classification(args=cla_args,
                                                                                list_single_part=list_single_data,
                                                                                multi_part=multi_data)
        else:
            list_single_data = cal_correlation(file_args)
            cf_list_single_model = process_CFDTAN(args=cf_args, dataset_name=file_args.dataset_name,
                                                  list_single_part=list_single_data)
            process_sax_slide(args=sax_slide_args, list_single_part=list_single_data)
            cla_list_single_model = process_classification(args=cla_args, list_single_part=list_single_data)
