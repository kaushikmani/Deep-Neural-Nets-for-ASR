import torch
import tables
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, file_name):
        super(Dataset, self).__init__()
        self.data = tables.open_file(file_name).root
        self.feature_seq = self.data['features']['feature']
        self.label_seq = self.data['labels']['feature']
        self.total_len = len(self.feature_seq)


    def __getitem__(self, index):
        feature_seq = self.feature_seq[index][0][0]
        label_seq = self.label_seq[index][0][0]
        return feature_seq, label_seq

    def __len__(self):
        return self.total_len
