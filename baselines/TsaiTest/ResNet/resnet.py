from tsai.basics import *
from tsai.models.ResNetPlus import *
from tsai.models.ResNet import *
from tsai.inference import load_learner
import torch
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import json


def train_resnet(dataset_name: str):
    print(f"{dataset_name} begin well done!")
    X, y, splits = get_classification_data(dsid=dataset_name, path='/home/zju/CFdtan',
                                           parent_dir='examples/data/npz/Univariate_npz',
                                           split_data=False,
                                           # verbose=True
                                           )
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_var=True)
    opt_func = wrap_optimizer(torch.optim.Adam)
    # clf = TSClassifier(X, y, splits=splits, path='./models/', arch=ResNetPlus,
    #                    # arch_config=dict(dropout=0.3, fc_dropout=0.9),
    #                    # arch_config=dict(dropout=0.3),
    #                    # arch_config=dict(fc_dropout=0.9),
    #                    arch_config=dict(ks=[8, 5, 3], fc_dropout=0.3),
    #                    # arch_config=dict(fc_dropout=0.3),
    #                    tfms=tfms, batch_tfms=batch_tfms, bs=128,
    #                    opt_func=opt_func,
    #                    # loss_func=LabelSmoothingCrossEntropyFlat(),
    #                    metrics=accuracy)
    clf = TSClassifier(X, y, splits=splits, path='./models/', arch=ResNet,
                       # arch_config=dict(dropout=0.3, fc_dropout=0.9),
                       # arch_config=dict(dropout=0.3),
                       # arch_config=dict(kss=[8, 5, 3]),
                       # arch_config=dict(fc_dropout=0.3),
                       tfms=tfms, batch_tfms=batch_tfms, bs=128,
                       opt_func=opt_func,
                       # loss_func=LabelSmoothingCrossEntropyFlat(),
                       metrics=accuracy)
    # clf.lr_find()
    clf.fit_one_cycle(200, 1e-3)
    # clf.plot_metrics()
    # clf.export("clf.pkl")

    probas, target, preds = clf.get_X_preds(X[splits[1]], y[splits[1]])
    true_preds = torch.argmax(probas, dim=1)
    acc = round(accuracy_score(target, true_preds), 4)
    print(f'accuracy: {acc}')
    print(f"{dataset_name} well done!")
    return acc


def test_tst(dataset_name: str):
    mv_clf = load_learner("models/clf.pkl")
    X, y, splits = get_classification_data(dsid=dataset_name, path='../..',
                                           parent_dir='examples/data/npz/Multivariate_npz',
                                           split_data=False, on_disk=False)
    probas, target, preds = mv_clf.get_X_preds(X[splits[1]], y[splits[1]])
    true_preds = torch.argmax(probas, dim=1)
    accuracy = accuracy_score(target, true_preds)
    print(f'accuracy: {accuracy:.4f}')
    pass


if __name__ == '__main__':
    # dataset_list = [
    #     'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
    #     'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
    #     'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration',
    #     'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
    #     'CricketZ', 'Crop', 'DiatomSizeReduction',
    #     'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
    #     'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
    #     'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
    #     'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal',
    #     'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
    #     'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain',
    #     'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
    #     'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan',
    #     'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
    #     'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
    #     'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
    #     'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
    #     'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
    #     'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
    #     'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
    #     'MoteStrain', 'NonInvasiveFetalECGThorax1',
    #     'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
    #     'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ',
    #     'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',
    #     'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
    #     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
    #     'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
    #     'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
    #     'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
    #     'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
    #     'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
    #     'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
    #     'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
    #     'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
    #     'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
    # ]
    dataset_list = ['GesturePebbleZ1']
    accuracy_list = []
    time_list = []
    for dataset in dataset_list:
        begin_time = time.time()
        try:
            acc = train_resnet(dataset)
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
    record_df.to_csv('./performance/resnet_final_record.csv', index=False)
