import numpy as np
import torch
from sklearn.metrics import accuracy_score

from data_helper.UEA_loader import get_UEA_data, processed_UEA_data
from shapelet.shapelet_discovery import shapelet_aggr_sax_and_slide
import json
from CFSC import CFDTAN_arg_parser, file_arg_parser, sax_slide_arg_parser
from uni_lstm import BiLSTMAttentionClassifier
from multi_gcn import cal_adj, transfer_inputs_sim
import pandas as pd

if __name__ == '__main__':

    # ["AtrialFibrillation","Handwriting","Libras", "FingerMovements","ERing","UWaveGestureLibrary","FingerMovements","LSST","NATOPS","Libras","ERing","MotorImagery","PhonemeSpectra","EthanolConcentration"]
    # 1、加载数据
    dataset_name = "FingerMovements"
    torch.cuda.empty_cache()
    data_dir = "../examples/data/arff/Multivariate_arff"
    X_train, X_test, y_train, y_test = processed_UEA_data(data_dir, dataset_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 2、根据数据属性加载模型类型
    # 1）、查看文件对于多元数据需要查看哪些数据作多元处理，哪些作单元处理
    channel_index_dir = "../channel_index/" + dataset_name + ".json"
    with open(channel_index_dir, 'r') as file:
        data = file.read()
        json_data = json.loads(data)
    channel_single = json_data["single_index"]
    channel_multi = json_data["multi_index"]

    # 2）、加载模型
    print("#####load model begin....#####")
    model_dir = "../checkpoints/"
    if len(channel_multi) > 0:
        multi_dir = model_dir + "classification/multiple/"
        multi_classifier = torch.load(multi_dir + dataset_name + '_classification_multi_checkpoint.pth')
        multi_classifier.eval()
    if len(channel_single) > 0:
        single_dir = model_dir + "classification/single/"
        single_classifier = [torch.load(single_dir + dataset_name + '_classification_single_' +
                                        str(i) + '_checkpoint.pth').eval()
                             for i in range(len(channel_single))]
    print("####load model end....####")
    # 3、测试效果评估
    y_pred = []
    cf_args, unknown_arg = CFDTAN_arg_parser()
    file_args, unknown_arg1 = file_arg_parser(unknown_arg)
    sax_slide_args, unknown_arg2 = sax_slide_arg_parser(unknown_arg1)

    for i in range(len(X_test)):
        print("turn i: {}, total: {}".format(i + 1, len(X_test)))
        x = np.expand_dims(X_test[i, :, :], axis=1)
        if len(channel_single) > 0:
            single_y_pre = []
            c = 0
            for j in channel_single:
                test_x = np.expand_dims(x[j], axis=0)
                test_y = np.expand_dims(y_test[j], axis=0)
                dict_slide, dict_dim, dict_class_label = shapelet_aggr_sax_and_slide(test_x, test_y,
                                                                                 sax_slide_args, False)
                total_single_x_list = []
                for key in dict_slide.keys():
                    total_single_x_list.extend(dict_slide[key].tolist())
                lengths = [len(seq) for seq in total_single_x_list]
                lengths = torch.tensor(lengths)
                max_length = torch.max(lengths)
                # Pad sequences with zeros to match the maximum length
                padded_sequences = [seq + [0] * (max_length - length) for seq, length in zip(total_single_x_list, lengths)]
                # # SVM模型输入需要ndarray
                # y = clf.predict(padded_sequences)
                # BiLSTM 的模型
                inputs = torch.tensor(padded_sequences)
                inputs = inputs.to(device)
                classifier = single_classifier[c]
                c = c+1
                y_out = classifier(inputs, lengths)
                _, pred = torch.max(y_out, dim=1)
                pred_arr = pred.cpu().numpy()
                single_y_pre.extend(pred_arr)


        if len(channel_multi) > 0:
            total_multi_x_list = []
            x = np.expand_dims(X_test[i, :, :], axis=0)
            test_x = x[:, channel_multi, :]
            dict_slide, dict_dim, dict_class_label, max_len, num_channels = shapelet_aggr_sax_and_slide(test_x, y_test,
                                                                                                       sax_slide_args, True)
            for key in dict_slide.keys():
                total_multi_x_list.extend(dict_slide[key].tolist())

            inputs, seq_len = transfer_inputs_sim(total_multi_x_list, max_len, num_channels)
            # Pad sequences with zeros to match the maximum length
            adj = cal_adj(inputs)
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            adj = torch.tensor(adj, dtype=torch.float32).to(device)
            # inputs = torch.tensor(inputs, dtype=torch.float32)
            # adj = torch.tensor(adj, dtype=torch.float32)
            seq_len = torch.tensor(seq_len, dtype=torch.int64)
            y_out = multi_classifier(inputs, adj, seq_len)
            _, pred = torch.max(y_out, dim=1)
            pred_arr = pred.cpu().numpy()
            single_y_pre.extend(pred_arr)
        # 获取数组中每个元素的唯一值和计数
        unique_values, counts = np.unique(single_y_pre, return_counts=True)
        # 找到计数数组中的最大值索引
        max_count_index = np.argmax(counts)
        most_pred = unique_values[max_count_index]
        y_pred.append(most_pred)


    if len(y_pred) > 0:
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_test, y_pred)
        print("dataset_name:{} accuracy:{}".format(dataset_name, acc))
