import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.multi_dataset import MultiDataset

import numpy as np


class GCLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, output_dim, num_layers):
        super(GCLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gcn = GCN(input_dim, hidden_dim, feature_dim, bias=True)
        # self.lstm = LSTM(hidden_dim, hidden_dim, num_layers)
        self.lstm = nn.GRU(hidden_dim, hidden_dim, num_layers)
        # fixme: the input size here is not determined by hidden_dim * input_dim, but hidden_dim * num_layers
        # self.fc = nn.Linear(hidden_dim * input_dim, output_dim)
        self.fc = nn.Linear(hidden_dim * num_layers, output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj, seq_len):
        gcn_out = self.gcn(x, adj)
        gcn_out = F.relu(gcn_out)
        # gcn_out = gcn_out.transpose(0, 1)  # 调整维度顺序，将时间维度放在第1维
        packed_embed = pack_padded_sequence(gcn_out, seq_len, batch_first=True, enforce_sorted=False)
        out, hidden = self.lstm(packed_embed.float())
        outputs, _ = pad_packed_sequence(out, batch_first=True)
        # hidden_out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # bug fix for cat computation
        hidden_out = torch.cat([hidden[i, :, :] for i in range(hidden.shape[0])], dim=1)
        hidden_out = self.dropout(hidden_out)
        hidden_out = self.fc(hidden_out)
        probabilities = self.softmax(hidden_out)
        return probabilities


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, bias=True):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain("tanh"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = x.transpose(1, 2)
        # support = torch.matmul(x, self.weight.double())
        # output = torch.matmul(adj.double(), support)

        support = torch.matmul(x, adj)
        output = torch.matmul(support, self.weight)

        if self.bias is not None:
            output = output + self.bias
        return output


# def calculate_laplacian_with_self_loop(matrix):
#     matrix = matrix + torch.eye(matrix.size(0))
#     row_sum = matrix.sum(1)
#     d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
#     d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
#     d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
#     normalized_laplacian = (
#         matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
#     )
#     return normalized_laplacian

def train_multi_classifier_sim(data, y, num_class, max_len, num_channels, args: argparse.Namespace):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inputs, seq_len = transfer_inputs_sim(data, max_len, num_channels)

    input_dim = inputs.shape[1]
    feature_dim = inputs.shape[2]
    adj = cal_adj(inputs)

    model = GCLSTM(input_dim, args.multi_hidden_dim, feature_dim, num_class, args.multi_num_layers).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.cla_lr)

    # adjust the length to inputs[B C L]'B
    y = y[::num_channels]


    inputs = torch.tensor(inputs, dtype=torch.float32)
    adj = torch.tensor(adj, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.long)
    seq_len = torch.tensor(seq_len, dtype=torch.int64)

    multi_dataset = MultiDataset(inputs, adj, targets, seq_len)
    multi_dataloader = DataLoader(multi_dataset, batch_size=args.cla_batch_size, shuffle=True)
    # 模型训练
    for epoch in tqdm(range(args.cla_n_epochs)):
        total = .0
        times = 0
        mean_acc = .0
        for batch in multi_dataloader:
            data, adj, target, seq_len = batch
            data, adj, target = data.to(device), adj.to(device), target.to(device)
            output = model(data, adj, seq_len)

            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * data.size(0)
            _, predicted = torch.max(output, dim=1)
            y_pred = predicted.cpu().numpy()
            target_batch = target.cpu().numpy()
            acc = accuracy_score(target_batch, y_pred)
            mean_acc += acc
            times += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {round(total / len(multi_dataset), 4)}, "
                  f"Accuracy: {round(mean_acc / times, 4)}")

    return model


def train_multi_classifier(inputs, targets,
                           num_classes,  # 分类类别数量
                           seq_num,  # 时间序列数量
                           max_slide_len,  # 最长子序列长度
                           hidden_dim=20,  # 隐藏层维度
                           num_layers=2,  # 模型层数
                           ):
    # transfer inputs & calculate mask
    inputs, mask, seq_len = transfer_inputs(inputs, max_slide_len)
    # 变量的维度
    input_dim = mask.shape[1]

    # 每条时间序列的长度
    feature_dim = mask.shape[2]
    # calculate adj
    adj = cal_adj(inputs)
    # adj = adj * mask
    model = GCLSTM(input_dim, hidden_dim, feature_dim, num_classes, num_layers)  # 输出维度改为类别数量
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    multi_dataset = MultiDataset(inputs, adj, targets, seq_len)
    multi_dataloader = DataLoader(multi_dataset, batch_size=100, shuffle=True)
    # 模型训练
    num_epochs = 1000
    for epoch in range(num_epochs):
        predicted_target = []
        total = 0
        for batch in multi_dataloader:
            data, adj, target, seq_len = batch
            output = model(torch.Tensor(data), adj, seq_len)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
            print("epoch: %s, loss: %s" % (epoch, loss.item() / data.shape[0]))
            # 计算准确率
            _, predicted = torch.max(output, 1)
            predicted_target = np.concatenate((predicted_target, predicted.numpy()), axis=0)
        accuracy = accuracy_score(targets, predicted_target)
        # accuracy = (predicted_target == target).sum().item() / seq_num

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")

    # 模型推断
    # with torch.no_grad():
    #     test_output = model(inputs, adj, mask)
    #     _, predicted = torch.max(test_output, 1)
    #     print("Test Predictions:", predicted)


def transfer_inputs_sim(inputs, max_slide_len, num_channels):
    # Pad sequences with zeros to match the maximum length
    seq_lens = [len(sublist) for sublist in inputs]
    extend_list = [sublist + [0] * (max_slide_len - seq_len) if seq_len < max_slide_len else sublist for
                   sublist, seq_len in zip(inputs, seq_lens)]
    seq_num = len(inputs)
    transferred_inputs = np.reshape(extend_list, (int(seq_num / num_channels), num_channels, max_slide_len))
    seq_lens = seq_lens[::num_channels]

    return transferred_inputs, seq_lens


# 转换inputs为同一长度
def transfer_inputs(inputs, max_slide_len):
    # Pad sequences with zeros to match the maximum length
    # fixme: there some bugs remain in multi-part
    seq_lens = [len(sublist) for sublist in inputs]
    extend_list = [sublist + [0] * (max_slide_len - seq_len) if seq_len < max_slide_len else sublist for
                   sublist, seq_len in zip(inputs, seq_lens)]
    # flattened_list = [item for sublist in extend_list for item in sublist]
    seq_num = len(inputs)
    channel = len(inputs[0])
    # transferred_inputs = np.reshape(flattened_list, (seq_num, channel, max_slide_len)).squeeze()
    transferred_inputs = np.reshape(extend_list, (seq_num, channel, max_slide_len)).squeeze()
    # Convert to PyTorch tensor
    # transferred_inputs = torch.tensor(transferred_inputs)
    # Create a mask tensor based on non-zero elements in inputs
    mask = (transferred_inputs != 0).float()

    return transferred_inputs, mask, seq_lens


# 计算每条数据的协方差系数
def cal_adj(inputs):
    adj = []
    num_channels = inputs.shape[1]
    for i in range(inputs.shape[0]):
        adjacency_matrix = np.abs(np.corrcoef(inputs[i]))
        adj.append(adjacency_matrix)
    seq_num = len(inputs)
    adj = np.reshape(adj, (seq_num, num_channels, num_channels))
    return adj


if __name__ == '__main__':

    # num_nodes = 10  # 节点数(时序的长度)
    # input_dim = 5  # 输入维度
    # num_classes = 3  # 分类类别数量
    # seq_num = 20  # 时间序列数量
    #
    # # todo: 修改成不同长度的list拼接
    # inputs = torch.randn(seq_num, num_nodes, input_dim)
    # adj = torch.randn(num_nodes, num_nodes)
    # targets = torch.randint(0, num_classes, (seq_num,))
    #
    # train_multi_classifier(inputs, targets, adj, input_dim, num_classes, seq_num)
    # 测试协方差
    # inputs = [[[1, 2, 3, 4, 5],[5, 4, 3, 2, 1]],[[1, 2, 5, 3, 5],[5, 4, 3, 2, 1]]]
    # adj = cal_adj(np.array(inputs), mask=None)

    # 真实数据
    from shapelet.shapelet_discovery_multi import shapelet_after_sax_and_slide
    from data_helper.UEA_loader import get_UEA_data

    data_dir = "../examples/data"
    dataset_name = "Libras"
    train_dataloader, validation_dataloader, test_dataloader, X_train, X_test, y_train, y_test = get_UEA_data(data_dir,
                                                                                                              dataset_name)
    dict_slide, dict_dim, dict_class_label, max_slide_len = shapelet_after_sax_and_slide(X_train, y_train)

    total_x_list = []
    total_y_list = []
    max_val = -np.inf
    for key in dict_slide.keys():
        tmp_max_val = np.amax(dict_slide[key])
        max_val = tmp_max_val if tmp_max_val > max_val else max_val
        total_x_list.extend(dict_slide[key])
    for key in dict_class_label.keys():
        total_y_list.extend(dict_class_label[key].astype('int'))
    num_classes = len(set(total_y_list))

    train_multi_classifier(total_x_list, total_y_list, num_classes, len(total_y_list), max_slide_len)
    pass
