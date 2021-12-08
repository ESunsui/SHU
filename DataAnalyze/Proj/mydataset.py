from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_X, data_y):
        self.data_X = data_X
        self.data_y = data_y

    def __getitem__(self, idx):
        return self.data_X.iloc[idx, :].values, self.data_y[idx]

    def __len__(self):
        return self.data_X.shape[0]
