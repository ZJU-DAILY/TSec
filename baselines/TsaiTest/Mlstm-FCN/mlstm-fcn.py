from tsai.basics import *
from tsai.models.RNN_FCNPlus import MLSTM_FCNPlus
# from tsai.inference import load_learner
import torch
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import json
import math


def train_mlstm_fcn(dataset_name: str):
    print(f"{dataset_name} begin well done!")
    X, y, splits = get_classification_data(dsid=dataset_name, path='/home/zju/CFdtan',
                                           parent_dir='examples/data/npz/Multivariate_npz',
                                           split_data=False)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_var=True)
    clf = TSClassifier(X, y, splits=splits, path='./models/', arch=MLSTM_FCNPlus,
                       # arch_config=dict(dropout=0.3, fc_dropout=0.9),
                       # arch_config=dict(dropout=0.3),
                       # todo: 此处需修改hidden_size 按原论文的意思
                       arch_config=dict(kss=[8, 5, 3]),
                       tfms=tfms, batch_tfms=batch_tfms, bs=128,
                       # loss_func=LabelSmoothingCrossEntropyFlat(),
                       metrics=accuracy)
    # metrics=accuracy)
    # clf.lr_find()
    clf.fit_one_cycle(100, 1e-3)
    clf.fit_one_cycle(100, 1e-3 / math.pow(2, 1 / 3))
    clf.fit_one_cycle(50, 1e-3 / (math.pow(2, 1 / 3) ** 2))
    # clf.plot_metrics()
    # clf.export("clf.pkl")

    probas, target, preds = clf.get_X_preds(X[splits[1]], y[splits[1]])
    true_preds = torch.argmax(probas, dim=1)
    acc = round(accuracy_score(target, true_preds), 4)
    print(f'accuracy: {acc}')
    print(f"{dataset_name} well done!")
    return acc


# def test_tst(dataset_name: str):
#     mv_clf = load_learner("models/clf.pkl")
#     X, y, splits = get_classification_data(dsid=dataset_name, path='../..',
#                                            parent_dir='examples/data/npz/Multivariate_npz',
#                                            split_data=False, on_disk=False)
#     probas, target, preds = mv_clf.get_X_preds(X[splits[1]], y[splits[1]])
#     true_preds = torch.argmax(probas, dim=1)
#     accuracy = accuracy_score(target, true_preds)
#     print(f'accuracy: {accuracy:.4f}')
#     pass


if __name__ == '__main__':
    # dataset_list = [
    #     'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
    #     'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
    #     'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
    #     'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    #     'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
    #     'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports',
    #     'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits',
    #     'StandWalkJump', 'UWaveGestureLibrary'
    # ]
    dataset_list = ['CharacterTrajectories']
    accuracy_list = []
    time_list = []
    for dataset in dataset_list:
        begin_time = time.time()
        try:
            acc = train_mlstm_fcn(dataset)
        except Exception as e:
            print(e)
            print(f"{dataset} has error!")
            acc = .0
        end_time = time.time()
        accuracy_list.append(acc)
        exe_time = round(end_time - begin_time, 1)
        temp_dict = {"dataset": dataset, "acc": acc, "exe_time": exe_time}
        with open(f'./performance/{dataset}.json', 'w') as f:
            json.dump(temp_dict, f)
        time_list.append(exe_time)
    record_dict = {"数据集名": dataset_list,
                   "准确率": accuracy_list,
                   "总耗时": time_list}
    record_df = pd.DataFrame(record_dict)
    record_df.to_csv('./performance/final_record.csv', index=False)
