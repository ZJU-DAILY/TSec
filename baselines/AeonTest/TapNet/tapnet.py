import json
import logging
import math
import time

import numpy as np
# from keras.optimizers.optimizer_v1 import Adam
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from tensorflow import keras

from aeon.classification.deep_learning.tapnet import TapNetClassifier
from aeon.datasets import load_classification


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class LossThresholdEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', threshold=1e-9):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.threshold = threshold
        self.prev_loss = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if self.prev_loss is not None and abs(current_loss - self.prev_loss) < self.threshold:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        self.prev_loss = current_loss


def train_tapnet(dataset_name: str):
    X_train, y_train = load_classification(name=dataset_name, split="train", return_metadata=False,
                                           extract_path="/home/zju/CFdtan/examples/data/ts/Multivariate_ts")
    if isinstance(X_train, np.ndarray):
        input_channel = X_train.shape[1]
    else:  # list of np array
        input_channel = X_train[0].shape[1]
    rp_params = (3, math.floor(input_channel * 1.5 / 3))
    optimizer = keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-3)
    # 创建 LossThresholdEarlyStopping 实例
    early_stop = LossThresholdEarlyStopping(monitor='loss', threshold=1e-9)
    model = TapNetClassifier(rp_params=rp_params, dilation=1, filter_sizes=(256, 256, 128), kernel_size=(8, 5, 3),
                             callbacks=early_stop, activation='softmax', loss=categorical_crossentropy, n_epochs=3000,
                             batch_size=16, optimizer=optimizer,
                             # verbose=True
                             )
    model.fit(X_train, y_train)

    X_test, y_test = load_classification(name=dataset_name, split="test", return_metadata=False,
                                         extract_path="/home/zju/CFdtan/examples/data/ts/Multivariate_ts")
    acc = model.score(X_test, y_test)
    return acc


if __name__ == '__main__':
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.WARN)
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
    # import tensorflow as tf

    # tf.config.list_physical_devices('GPU')
    # import tensorflow as tf
    #
    # build = tf.sysconfig.get_build_info()
    # print(build['cuda_version'])
    # print(build['cudnn_version'])
    #
    # import torch
    # print(torch.__version__)  # torch version
    # print(torch.version.cuda)  # cuda version
    # print(torch.backends.cudnn.version())  # cudnn version

    # dataset_list = ['InsectWingbeat', 'JapaneseVowels', 'SpokenArabicDigits']
    dataset_list = ['AtrialFibrillation']
    accuracy_list = []
    time_list = []
    for dataset in dataset_list:
        begin_time = time.time()
        try:
            acc = train_tapnet(dataset)
        except Exception as e:
            print(e)
            print(f"{dataset} has error!")
            acc = .0
        end_time = time.time()
        accuracy_list.append(acc)
        exe_time = round((end_time - begin_time) / 60, 2)
        temp_dict = {"dataset": dataset, "acc": acc, "exe_time": exe_time}
        with open(f'/home/zju/CFdtan/baselines/AeonTest/TapNet/performance/{dataset}.json', 'w') as f:
            json.dump(temp_dict, f)
        time_list.append(exe_time)
    # record_dict = {"数据集名": dataset_list,
    #                "准确率": accuracy_list,
    #                "总耗时": time_list}
    # record_df = pd.DataFrame(record_dict)
    # record_df.to_csv('/home/zju/CFdtan/baselines/AeonTest/TapNet/performance/add_record.csv', index=False)
