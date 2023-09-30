import numpy as np


def loss_dict(name):
    # loss_dict = {
    #     'train_type': name,
    #     'total_loss': [],
    #     'alignment_loss': [],
    #     'smoothness_prior_loss': [],
    #     'epoch': []
    # }
    return {
        'train_type': name,
        'total_loss': [],
        'alignment_loss': [],
        'smoothness_prior_loss': [],
        'epoch': []
    }


class ExperimentClass:
    def __init__(self, n_epochs, batch_size, lr, exp_name, device='cpu'):
        self.cf_args = None
        self.model = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.exp_name = exp_name
        self.device = device
        self.cf_network = None

        self.loss_tracker = {
            'train': loss_dict(name='train'),
            'validation': loss_dict(name='validation'),
            'test': loss_dict(name='test')
        }

    def __str__(self):
        return str(self.__dict__)

    def update_loss(self, phase, epoch, alignment_loss, smoothness_prior_loss):
        total_loss = alignment_loss + smoothness_prior_loss
        self.loss_tracker[phase]['total_loss'].append(total_loss)
        self.loss_tracker[phase]['alignment_loss'].append(alignment_loss)
        self.loss_tracker[phase]['smoothness_prior_loss'].append(smoothness_prior_loss)
        self.loss_tracker[phase]['epoch'].append(epoch)

    def get_min_loss(self, phase='train', loss_type='total_loss', return_epoch=False):
        loss_arr = np.asarray(self.loss_tracker[phase][loss_type])
        min_loss = np.amin(loss_arr)
        if not return_epoch:
            return min_loss
        else:
            min_loss_idx = loss_arr.argmin()
            epoch = self.loss_tracker[phase]['epoch'][min_loss_idx]
            return min_loss, epoch

    def print_min_loss_all(self):
        print("--- Printing minimum loss --")
        for phase in ["train", "validation", "test"]:
            total_loss, min_epoch = self.get_min_loss(phase, "total_loss", True)
            print(f"{phase} minimum (total) loss: {total_loss}")

    def add_cf_args(self, cf_args):
        self.cf_args = cf_args

    def get_cf_args(self):
        return self.cf_args

    def add_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def add_cf_network(self, cf_net):
        self.cf_network = cf_net

    def get_cf_network(self):
        return self.cf_network


class ExperimentsManager:
    def __init__(self):
        self.experiments_dict = {}

    def add_experiment(self, exp_name, n_epochs, batch_size, lr, device):
        self.experiments_dict[exp_name] = ExperimentClass(n_epochs, batch_size, lr, exp_name, device)

    def get_experiment(self, exp_name):
        return self.__getitem__(exp_name)

    def __getitem__(self, exp_name):
        return self.experiments_dict[exp_name]

    def __str__(self):
        return str(self.__dict__)


class CFArgs:
    def __init__(self, tess_size=32, smoothness_prior=True, lambda_smooth=1, lambda_var=0.1, n_recurrences=1,
                 zero_boundary=True, T=None, n_ss=0, back_version=True):
        self.tess_size = tess_size
        self.smoothness_prior = smoothness_prior
        self.lambda_smooth = lambda_smooth
        self.lambda_var = lambda_var
        self.n_recurrences = n_recurrences
        self.zero_boundary = zero_boundary
        self.n_ss = n_ss
        self.back_version = back_version
        self.T = T

    def __str__(self):
        return str(self.__dict__)

    def set_basis(self, T):
        self.T = T


def max_in_nested_list(nested_list):
    """
    递归求解多级列表的最大值
    """
    # 如果列表为空，返回 None
    if not nested_list:
        return None

    # 假设第一个元素是最大值
    max_val = nested_list[0]

    for elem in nested_list:
        # 如果当前元素是列表类型，递归调用该函数进行查找
        if isinstance(elem, list):
            val = max_in_nested_list(elem)
        else:
            val = elem

        # 更新最大值
        if val > max_val:
            max_val = val

    return max_val
