import torch
from difw import Cpab
from torch import nn


# def get_loc_net_part1():
#     part1 = nn.Sequential(
#         nn.Conv1d(1, 128, kernel_size=7),
#         nn.BatchNorm1d(128),
#         nn.MaxPool1d(3, stride=2),
#         nn.ReLU(True)
#     )
#
#     return part1


# todo: 按照原网络 等于depth=3 这样网络在全连接层是不是过于冗余 故考虑将depth减少为2
def temp_get_loc_net_part1(n_channels=1):
    temp_loc_net_part1 = nn.Sequential(
        nn.Conv1d(n_channels, 128, kernel_size=3),
        nn.BatchNorm1d(128),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True),

        nn.Conv1d(128, 64, kernel_size=5),
        nn.BatchNorm1d(64),
        nn.MaxPool1d(3, stride=3),
        nn.ReLU(True),

        nn.Conv1d(64, 64, kernel_size=3),
        nn.BatchNorm1d(64),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True)
    )
    return temp_loc_net_part1


def temp_get_loc_net_part2(input_dim):
    temp_loc_net_part2 = nn.Sequential(
        nn.Linear(input_dim, 48),
        nn.ReLU(True),

        nn.Linear(48, 32),
        nn.ReLU(True),

        nn.Linear(32, 16),
        nn.ReLU(True)
    )

    return temp_loc_net_part2


class CfDtAn(nn.Module):
    def __init__(self, signal_len, channels, tess_size=6, n_recur=1, zero_boundary=True, device='gpu', num_scaling=0,
                 back_version=True):
        super(CfDtAn, self).__init__()
        if zero_boundary:
            self.T = Cpab(tess_size=tess_size, backend='pytorch', device=device, zero_boundary=zero_boundary,
                          basis='qr')
        else:
            self.T = Cpab(tess_size=tess_size - 1, backend='pytorch', device=device, zero_boundary=zero_boundary,
                          basis='qr')
        self.dim = tess_size - 1 if zero_boundary else tess_size
        self.n_recurrence = n_recur
        self.input_shape = signal_len
        self.channels = channels
        self.num_scaling = num_scaling
        self.net_part1 = temp_get_loc_net_part1(n_channels=self.channels)
        self.back_version = back_version
        self.fc_input_dim1 = self.get_conv_to_fc_dim()

        self.net_part2 = temp_get_loc_net_part2(self.fc_input_dim1)

        # self.net_mc_part3 = nn.ModuleList()
        #
        # for i in range(self.channels):
        #     fc_s = nn.Sequential(
        #         nn.Linear(16, self.dim),
        #         nn.Tanh()
        #     )
        #     self.net_mc_part3.append(fc_s)

        if not self.back_version:
            self.net_mc_part3 = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(16, self.dim),
                    nn.Tanh()
                )
                for _ in range(self.channels)
            ])

            for fc_s in self.net_mc_part3:
                fc_s[0].weight.data.zero_()
                fc_s[-2].bias.data.copy_(torch.clone(self.T.identity(epsilon=0.001).view(-1)))
        else:
            self.net_part3 = nn.Sequential(
                nn.Linear(16, self.dim),
                nn.Tanh()
            )
            self.net_part3[0].weight.data.zero_()
            self.net_part3[-2].bias.data.copy_(torch.clone(self.T.identity(epsilon=0.001).view(-1)))

    def get_conv_to_fc_dim(self):
        rand_tensor = torch.rand([1, self.channels, self.input_shape])
        out_tensor = self.net_part1(rand_tensor)

        # todo: 这里需验证如何设计最后的线性层能使模型的效果变好
        # 需不需要将倒数第二部分也变成各通道独立的值得进一步讨论
        return out_tensor.size(2) if not self.back_version else out_tensor.size(1) * out_tensor.size(2)

    def stn(self, x, return_theta=False):
        xs = self.net_part1(x)
        if not self.back_version:
            xs = self.net_part2(xs)
            thetas = []
            # if self.channels != 1:
            #     raise Exception("not completed yet for the multi-channel in one single run")
            for i in range(self.channels):
                feature_map = xs[:, i, :]
                # fixme: 修改linear输入层的大小
                flattened_map = feature_map.view(-1, 16)
                temp_theta = self.net_mc_part3[i](flattened_map)
                thetas.append(temp_theta)
            theta = torch.stack(thetas, dim=0)
            x = torch.transpose(x, 1, 2)
            xs = x.clone()
            for i in range(self.channels):
                xs[:, :, i] = torch.squeeze(
                    self.T.transform_data_ss(torch.unsqueeze(x[:, :, i], dim=-1), theta[i], outsize=self.input_shape,
                                             N=self.num_scaling), dim=-1)
            x = torch.transpose(xs, 2, 1)
        else:
            xs = xs.view(-1, self.fc_input_dim1)
            xs = self.net_part2(xs)
            theta = self.net_part3(xs)
            x = torch.transpose(x, 1, 2)
            x = self.T.transform_data_ss(x, theta, outsize=self.input_shape, N=self.num_scaling)
            x = torch.transpose(x, 2, 1)
        if not return_theta:
            return x
        else:
            return x, theta

    def forward(self, x, return_theta=False):
        thetas = []
        for i in range(self.n_recurrence):
            if not return_theta:
                x = self.stn(x)
            else:
                x, theta = self.stn(x, return_theta)
                thetas.append(theta)
        if not return_theta:
            return x
        else:
            return x, thetas

    def get_basis(self):
        return self.T
