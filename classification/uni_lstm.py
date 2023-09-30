import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

# from examples.UCR_alignment import run_UCR_alignment
# from examples.UCR_alignment import arg_parser, run_UCR_alignment
# from utils.time2vec import Time2Vec
from time2vec.Model import Time2Vec
from utils.uni_dataset import UniDataset

import numpy as np

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(BiLSTMAttentionClassifier, self).__init__()
        self.embedding = Time2Vec('sin', embedding_dim)
        self.bilstm = nn.GRU(2, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.fc = nn.Linear(hidden_dim * 2, num_classes, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # def forward(self, inputs: torch.Tensor, mask):
    def forward(self, inputs: torch.Tensor, lengths):
        embedded = self.embedding(torch.unsqueeze(inputs, dim=-1))
        packed_embed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        outputs, hidden = self.bilstm(packed_embed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        # attention_weights = self.attention(hidden)
        # attended_outputs = torch.sum(torch.mul(outputs, attention_weights), dim=1)
        # attended_outputs = torch.mul(hidden, attention_weights)
        attended_outputs = self.relu(hidden)
        logits = self.fc(attended_outputs)
        probabilities = self.softmax(logits)

        return probabilities


def train_uni_classifier_sim(data, y, num_class, args: argparse.Namespace, target_loss=0.01):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lengths = [len(seq) for seq in data]
    lengths = torch.tensor(lengths)
    max_length = torch.max(lengths)

    padded_sequences = [seq + [0] * (max_length - length) for seq, length in zip(data, lengths)]
    inputs = torch.tensor(padded_sequences)

    model = BiLSTMAttentionClassifier(args.uni_embedding_dim, args.uni_hidden_dim, num_class).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.cla_lr)
    targets = torch.LongTensor(y)

    uni_dataset = UniDataset(inputs, lengths, targets)
    uni_dataloader = DataLoader(uni_dataset, batch_size=args.cla_batch_size, shuffle=True)

    model.train()
    for i in tqdm(range(args.cla_n_epochs)):
        mean_loss = .0
        times = 0
        mean_acc = .0
        # Perform the forward pass and compute the loss
        for batch in uni_dataloader:
            input_batch, length_batch, target_batch, index_batch = batch
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            output = model(input_batch.float(), length_batch)
            # output = output.to(torch.int64)
            loss = loss_function(output, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item() * input_batch.size(0)
            _, y_pred = torch.max(output, dim=1)
            y_pred = y_pred.cpu().numpy()
            target_batch = target_batch.cpu().numpy()
            acc = accuracy_score(target_batch, y_pred)
            mean_acc += acc
            times += 1

        mean_loss /= len(uni_dataset)
        mean_acc /= times
        # Check if the loss is below the target loss
        if (i + 1) % 10 == 0:
            print("Iteration:", i + 1, "Loss: {:.4f}".format(mean_loss), "Accuracy: {:.4f}".format(mean_acc))
        # fixme: want something like early-stop, there are some other ways
        if mean_loss < target_loss:
            print("Reached target loss of", target_loss)
            break
        # if mean_acc == 1.0:
        #     break

    return model


def train_uni_classifier(inputs, targets, num_classes,
                         embedding_dim=50,
                         hidden_dim=128,
                         num_iterations=300,
                         target_loss=0.001):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # fixme: whether the original data need normalization, single or multi
    # # 对训练集数据进行标准化
    # # 计算每个子列表的均值和标准差
    # means = [np.nanmean(sublist) for sublist in inputs_train]
    # stds = [np.nanstd(sublist) for sublist in inputs_train]
    # # 对每个子列表进行标准化
    # inputs_arr = [(sublist - mean) / std if std != 0 else sublist for sublist, mean, std in zip(inputs_train, means, stds)]
    # inputs = [list(arr_data) for arr_data in inputs_arr]
    lengths = [len(seq) for seq in inputs]
    lengths = torch.tensor(lengths)
    max_length = torch.max(lengths)

    # Pad sequences with zeros to match the maximum length
    padded_sequences = [seq + [0] * (max_length - length) for seq, length in zip(inputs, lengths)]
    # Convert to PyTorch tensor
    inputs = torch.tensor(padded_sequences)

    # Create an instance of the classifier
    classifier = BiLSTMAttentionClassifier(embedding_dim, hidden_dim, num_classes).to(device)
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    targets = torch.tensor(targets)

    uni_dataset = UniDataset(inputs, lengths, targets)
    # uni_dataloader = DataLoader(uni_dataset, batch_size=64, shuffle=True)

    classifier.train()
    inputs_arr = inputs.cpu().numpy()
    lengths_arr = lengths.cpu().numpy()
    targets_arr = targets.cpu().numpy()
    # Perform multiple iterations of training until the loss is close to 0
    highest_acc = 0
    highest_inputs = []
    highest_targets = []
    for n in range(1):
        batch_size = 128
        uni_dataloader = DataLoader(uni_dataset, batch_size, shuffle=True)
        for i in range(num_iterations):
            mean_loss = .0
            times = 0
            mean_acc = .0
            if len(uni_dataset) == 0:
                break
            # uni_dataloader = DataLoader(uni_dataset, batch_size, shuffle=True)
            index_wrong_all = []
            # Perform the forward pass and compute the loss
            for batch in uni_dataloader:
                input_batch, length_batch, target_batch, index_batch = batch
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                output = classifier(input_batch.float(), length_batch)
                loss = loss_function(output, target_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
                _, y_pred = torch.max(output, dim=1)
                y_pred = y_pred.cpu().numpy()
                target_batch = target_batch.cpu().numpy()
                if i == num_iterations:
                    index_batch = index_batch.cpu().numpy()
                    index = np.arange(0, len(target_batch))
                    index_wrong = index[y_pred != target_batch]
                    index_wrong_batch = index_batch[index_wrong]
                    index_wrong_all = np.concatenate((index_wrong_all, index_wrong_batch), axis=0)
                acc = accuracy_score(target_batch, y_pred)
                mean_acc += acc
                times += 1

            if i == num_iterations:
                print("begin delete source data")
                index_wrong_all = index_wrong_all.astype(np.int32)
                inputs_arr = np.delete(inputs_arr, index_wrong_all, axis=0)
                lengths_arr = np.delete(lengths_arr, index_wrong_all, axis=0)
                targets_arr = np.delete(targets_arr, index_wrong_all, axis=0)
                uni_dataset = UniDataset(torch.Tensor(inputs_arr), lengths_arr, targets_arr)
            mean_loss /= times
            mean_acc /= times
            # Check if the loss is below the target loss
            print("Iteration:", i, "Loss:", mean_loss, "Accuracy:", mean_acc)
            if mean_acc > highest_acc and len(set(targets_arr)) == num_classes:
                highest_acc = mean_acc
                highest_inputs = inputs_arr
                highest_targets = targets_arr
                highest_lengths = lengths_arr
            if mean_loss < target_loss:
                print("Reached target loss of", target_loss)
                break
            if mean_acc == 1.0:
                break

    # np.savetxt('results/shapelets.csv', highest_inputs, delimiter=',')
    # np.savetxt('results/shapelets_label.csv', highest_targets, delimiter=',')
    # np.savetxt('results/shapelets_lengths.csv', highest_lengths, delimiter=',')

    return classifier


def train_svm(X_train, y_train, X_test, y_test):
    import pickle
    # t2vec = Time2Vec('sin', 320)
    lengths_train = [len(seq) for seq in X_train]
    lengths_train = torch.tensor(lengths_train)
    max_length_train = torch.max(lengths_train)
    # Pad sequences with zeros to match the maximum length
    padded_train = [seq + [0] * (max_length_train - length) for seq, length in zip(X_train, lengths_train)]
    # embedding_train = t2vec(torch.unsqueeze(torch.Tensor(padded_train), dim=-1))

    lengths_test = [len(seq) for seq in X_test]
    lengths_test = torch.tensor(lengths_test)
    max_length_test = torch.max(lengths_test)

    # Pad sequences with zeros to match the maximum length
    padded_test = [seq + [0] * (max_length_test - length) for seq, length in zip(X_test, lengths_test)]
    # embedding_test = t2vec(torch.unsqueeze(torch.Tensor(padded_test), dim=-1))
    # # 测试
    # with open('models/' + dataset_name + '_uni_model.pkl', 'rb') as f:
    #     clf = pickle.load(f)
    # y_predict = clf.predict(padded_test)
    # acc = accuracy_score(y_test, y_predict)
    # print(acc)

    clf = svm.SVC()
    # clf = xgb.XGBClassifier()
    print("model train begin...")
    # embedding_train.detach().numpy()
    clf.fit(padded_train, y_train)
    print("model train done...")
    y_predict = clf.predict(padded_test)
    acc = accuracy_score(y_test, y_predict)
    print(acc)
    # 保存模型
    with open('models/' + dataset_name + '_uni_model.pkl', 'wb') as f:
        # 创建空文件
        f.write(b'')

        # 将模型保存到文件
        pickle.dump(clf, f)
        print("save model done")


def test_svm(X_test, y_test):
    import pickle

    lengths_test = [len(seq) for seq in X_test]
    lengths_test = torch.tensor(lengths_test)
    max_length_test = torch.max(lengths_test)

    # Pad sequences with zeros to match the maximum length
    padded_test = [seq + [0] * (max_length_test - length) for seq, length in zip(X_test, lengths_test)]

    # 测试
    with open('models/' + dataset_name + '_uni_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    y_predict = clf.predict(padded_test)
    acc = accuracy_score(y_test, y_predict)
    print(acc)
    # 初始化一个字典来存储每个类别的分类正确数据的索引
    correct_data_indices = {}

    # 遍历预测结果和真实标签，找到每个类别分类正确的数据
    for label in np.unique(y_test):
        correct_indices = np.where((y_predict == label) & (y_predict == y_test))[0]
        correct_data_indices[label] = correct_indices

    # 打印每个类别分类正确的数据的索引
    for label, indices in correct_data_indices.items():
        print("Class:", label)
        print("Correct Data Indices:{},num:{}".format(indices, len(indices)))


if __name__ == '__main__':
    # 真实数据
    from data_helper.UCR_loader import processed_UCR_data
    from shapelet.shapelet_discovery import shapelet_return_sax_and_slide
    import numpy as np

    data_dir = "../examples/data"
    dataset_name = "Computers"

    # 获取数据
    X_train, X_test, y_train, y_test = processed_UCR_data(data_dir, dataset_name, suffix=True)

    # 数据对齐
    # args = arg_parser()
    # X_alied_train = run_UCR_alignment(args, dataset_name)
    #
    # print("train data process...")
    # # dict_slide, dict_dim, dict_class_label = shapelet_after_sax_and_slide(X_train, y_train)
    # dict_slide, dict_dim, dict_class_label, dict_sax_train, dict_index_train = \
    #     shapelet_return_sax_and_slide(X_alied_train, y_train)
    # print("train data process done...")
    # total_x_list = []
    # total_y_list = []
    # for key in dict_slide.keys():
    #     total_x_list.extend(dict_slide[key].tolist())
    # for key in dict_class_label.keys():
    #     total_y_list.extend(dict_class_label[key].astype('int').tolist())
    # # 获取数组中每个元素的唯一值和计数
    # unique_values, counts = np.unique(total_y_list, return_counts=True)
    # print("train unique_values:{},counts:{}".format(unique_values, counts))
    #
    # print("test data process...")
    # # dict_slide_test, dict_dim_test, dict_class_label_test = \
    # #     shapelet_after_sax_and_slide(X_test, y_test)
    # dict_slide_test, dict_dim_test, dict_class_label_test, dict_sax_test, dict_index_test = \
    #     shapelet_return_sax_and_slide(X_test, y_test)
    # print("test data process done...")
    # total_x_list_test = []
    # total_y_list_test = []
    # for key in dict_slide_test.keys():
    #     total_x_list_test.extend(dict_slide_test[key].tolist())
    # for key in dict_class_label_test.keys():
    #     total_y_list_test.extend(dict_class_label_test[key].astype('int').tolist())
    # # 获取数组中每个元素的唯一值和计数
    # unique_values_test, counts_test = np.unique(total_y_list_test, return_counts=True)
    # print("test unique_values:{},counts:{}".format(unique_values_test, counts_test))
    #
    # # test_svm(total_x_list_test, total_y_list_test)
    # # train_svm(total_x_list, total_y_list, total_x_list_test, total_y_list_test)
    #
    # num_classes = len(set(total_y_list))
    # classifier = train_uni_classifier(inputs=total_x_list, targets=total_y_list, num_classes=num_classes)
    # torch.save(classifier, 'models/' + dataset_name + '_uni_model.pth')
    # # pass
