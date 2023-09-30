from torch.utils.data import Dataset


class UniDataset(Dataset):
    def __init__(self, data, lengths, targets):
        self.data = data
        self.lengths = lengths
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx], self.targets[idx], idx
