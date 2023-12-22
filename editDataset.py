import torch
from torch.utils.data import Dataset, DataLoader

class EditDataset(Dataset):
    def __init__(self,edits,questions,nos):
        self.edits = edits
        self.questions = questions
        self.nos = nos

    def __getitem__(self, index):
        
        return self.edits[index],self.questions[index],self.nos[index]

    def __len__(self):
        return len(self.edits)


class EditSeqDataset(Dataset):
    def __init__(self,data):
        self.pairs = data

    def __getitem__(self, index):
        return self.pairs[index]["edit"],self.pairs[index]["questions"]

    def __len__(self):
        return len(self.pairs)