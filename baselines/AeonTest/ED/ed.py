import json
import time

import pandas as pd

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_classification


def train_ed(dataset_name: str):
    X_train, y_train = load_classification(name=dataset_name, split="train", return_metadata=False,
                                           # extract_path="/home/zju/CFdtan/examples/data/ts/Multivariate_ts"
                                           extract_path="/home/zju/CFdtan/examples/data/ts/Univariate_ts"
                                           )
    model = KNeighborsTimeSeriesClassifier(distance="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)

    X_test, y_test = load_classification(name=dataset_name, split="test", return_metadata=False,
                                         # extract_path="/home/zju/CFdtan/examples/data/ts/Multivariate_ts"
                                         extract_path="/home/zju/CFdtan/examples/data/ts/Univariate_ts"
                                         )
    acc = model.score(X_test, y_test)
    return acc


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
    dataset_list = [
        'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
        'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
        'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration',
        'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
        'CricketZ', 'Crop', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
        'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
        'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal',
        'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
        'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain',
        'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
        'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
        'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
        'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
        'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
        'MoteStrain', 'NonInvasiveFetalECGThorax1',
        'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
        'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ',
        'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',
        'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
        'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
        'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
        'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
        'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
        'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
        'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
    ]
    # dataset_list = ['AtrialFibrillation']
    accuracy_list = []
    time_list = []
    for dataset in dataset_list:
        begin_time = time.time()
        try:
            acc = train_ed(dataset)
        except Exception as e:
            print(e)
            print(f"{dataset} has error!")
            acc = .0
        end_time = time.time()
        accuracy_list.append(acc)
        exe_time = round((end_time - begin_time) / 60, 2)
        temp_dict = {"dataset": dataset, "acc": acc, "exe_time": exe_time}
        with open(f'/home/zju/CFdtan/baselines/AeonTest/ED/performance/{dataset}.json', 'w') as f:
            json.dump(temp_dict, f)
        time_list.append(exe_time)
    record_dict = {"数据集名": dataset_list,
                   "准确率": accuracy_list,
                   "总耗时": time_list}
    record_df = pd.DataFrame(record_dict)
    record_df.to_csv('/home/zju/CFdtan/baselines/AeonTest/ED/performance/ed_single_final_record.csv', index=False)
