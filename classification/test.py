import math

import numpy as np
from sklearn.metrics import accuracy_score

from data_helper.UCR_loader import processed_UCR_data


def cal_dis(X, candidate, candidate_length):
    feature = []
    total_index = []
    for i in range(np.shape(X)[0]):  # 遍历每个样本
        print("turn i:", i)
        temp_feat = []
        temp_index = []
        for j in range(len(candidate)):  # 遍历每个候选形状元素
            dist = math.inf
            index = -1
            candidate_tmp = np.asarray(candidate[j])  # 获取当前候选形状元素
            length = int(candidate_length[j])
            candidate_tmp = candidate_tmp[:length]
            for k in range(np.shape(X)[2] - np.shape(candidate_tmp)[0] + 1):  # 遍历样本中所有长度大于等于候选形状元素长度的子序列
                difference = X[i, :, 0 + k: int(np.shape(candidate_tmp)[0]) + k] - candidate_tmp  # 计算当前子序列与候选形状元素之间的距离
                feature_tmp = np.linalg.norm(difference)  # 计算欧氏距离
                if feature_tmp < dist:  # 更新距离
                    dist = feature_tmp
                    index = j
            temp_feat.append(dist)  # 将距离加入特征列表中
            temp_index.append(index)
        feature.append(min(temp_feat))
        min_index = temp_index[np.argmin(temp_feat)]
        total_index.append(min_index)
    return feature, total_index


if __name__ == '__main__':
    data_dir = "../examples/data"
    dataset_name = "ChlorineConcentration"
    X_train, X_test, y_train, y_test = processed_UCR_data(data_dir, dataset_name, suffix=True)
    # dict_slide, dict_dim, dict_class_label = shapelet_after_sax_and_slide(X_train, y_train)

    data_arr = np.loadtxt('results/shapelets.csv', delimiter=',')
    label_arr = np.loadtxt('results/shapelets_label.csv', delimiter=',')
    data_len_arr = np.loadtxt('results/shapelets_lengths.csv', delimiter=',')

    dists, indexes = cal_dis(X_train, data_arr, data_len_arr)
    predict = label_arr[indexes]
    acc = accuracy_score(y_train, predict)
    print(acc)

    # # 加载单元模型
    # modal = torch.load('uni_model.pth')
    # slide_size = 0.6
    # slide_step = 0.2
    # slide_num = 3
    # seq_len = X_test.shape[1]
    # if seq_len <= 50:
    #     step = 1
    # elif seq_len <= 100:
    #     step = 2
    # elif seq_len <= 300:
    #     step = 3
    # elif seq_len <= 1000:
    #     step = 4
    # elif seq_len <= 1500:
    #     step = 5
    # elif seq_len <= 2000:
    #     step = 7
    # elif seq_len <= 3000:
    #     step = 10
    # else:
    #     step = 100
    # for i in range(X_test.shape[0]):
    #     ts = X_test[i]
    #     for i in range(slide_num):
    #         slide_len = seq_len * (slide_size-slide_step*i)
    #         for k in range(1, seq_len - slide_len + 1, step):
    #             ts_temp = ts[:, :, k: slide_len + k]
    #             if isinstance(ts_temp, np.ndarray):
    #                 ts_alpha = np.concatenate((ts_alpha, ts_temp), axis=0)
    #             elif isinstance(ts_temp, torch.Tensor):
    #                 ts_alpha = torch.cat((ts_alpha, ts_temp), dim=0)
    #             else:
    #                 raise Exception("Unknown data type!")
