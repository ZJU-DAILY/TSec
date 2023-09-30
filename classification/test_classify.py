import numpy as np
import torch
from sklearn.metrics import accuracy_score

from data_helper.UCR_loader import processed_UCR_data
from shapelet.shapelet_discovery import shapelet_aggr_sax_and_slide
from CFSC import CFDTAN_arg_parser, file_arg_parser, sax_slide_arg_parser

if __name__ == '__main__':

    # ["ChlorineConcentration","ECGFiveDays","Computers", "Coffee"]
    # ["FacesUCR", "Ham", "Haptics", "inlineSkate", "InsectWingbeatSound"
    #     , "OSULeaf", "Phoneme", "RefrigerationDevices", "ShapeletSim", "UWaveGestureLibraryY", "WormsTwoClass"]
    dataset_name = "FiftyWords"
    torch.cuda.empty_cache()
    data_dir = "../examples/data/arff/Univariate_arff"
    X_train, X_test, y_train, y_test = processed_UCR_data(data_dir, dataset_name, suffix=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_dir = "../checkpoints/"
    single_dir = model_dir + "classification/single/"
    classifier = torch.load(single_dir + dataset_name + '_classification_single_0' + '_checkpoint.pth')
    classifier.eval()

    # 加载SVM训练的模型
    # with open('models/' + dataset_name + '_uni_model.pkl', 'rb') as f:
    #     clf = pickle.load(f)

    # >>> 单条数据测试 <<<
    cf_args, unknown_arg = CFDTAN_arg_parser()
    file_args, unknown_arg1 = file_arg_parser(unknown_arg)
    sax_slide_args, unknown_arg2 = sax_slide_arg_parser(unknown_arg1)
    max_val = -np.inf
    y_pred = []
    for i in range(len(X_test)):
        print("turn i: {}, total: {}".format(i + 1, len(X_test)))

        test_x = np.expand_dims(X_test[i], axis=0)
        test_y = np.expand_dims(y_test[i], axis=0)

        dict_slide, dict_dim, dict_class_label = shapelet_aggr_sax_and_slide(test_x, test_y,
                                                                             sax_slide_args, False)

        total_x_list = []
        output = []
        for key in dict_slide.keys():
            total_x_list.extend(dict_slide[key].tolist())
        if len(total_x_list) != 0:
            lengths = [len(seq) for seq in total_x_list]
            lengths = torch.tensor(lengths)
            max_length = torch.max(lengths)
            # Pad sequences with zeros to match the maximum length
            padded_sequences = [seq + [0] * (max_length - length) for seq, length in zip(total_x_list, lengths)]
            # # SVM模型输入需要ndarray
            # y = clf.predict(padded_sequences)
            # BiLSTM 的模型
            inputs = torch.tensor(padded_sequences)
            inputs = inputs.to(device)
            y_out = classifier(inputs, lengths)
            _, pred = torch.max(y_out, dim=1)
            pred_arr = pred.cpu().numpy()
            output.extend(pred_arr)
        # 获取数组中每个元素的唯一值和计数
        unique_values, counts = np.unique(output, return_counts=True)
        # 找到计数数组中的最大值索引
        max_count_index = np.argmax(counts)
        most_pred = unique_values[max_count_index]
        y_pred.append(most_pred)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("dataset_name:{} accuracy:{}".format(dataset_name, acc))
