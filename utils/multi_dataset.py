from torch.utils.data import Dataset


class MultiDataset(Dataset):
    def __init__(self, data, adj, targets, seq_len):
        self.data = data
        self.adj = adj
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.adj[idx], self.targets[idx], self.seq_len[idx]
