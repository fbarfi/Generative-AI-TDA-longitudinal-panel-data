import pytorch_lightning as pl
from dgl.dataloading import GraphDataLoader
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

from .dglgraphsdataset import DGLGraphsDataset

class GraphsDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args['batch_size']
        self.val_ratio = args['val_ratio']
        self.test_ratio = args['test_ratio']


    def setup(self, stage=None):
        self.dataset = DGLGraphsDataset(self.args)
        
        labels = self.dataset.labels
        num_graphs = len(self.dataset.graphs)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        self.label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))

        test_ratio = self.args.get("test_ratio", 0.1)
        val_ratio = self.args.get("val_ratio", 0.1)
        n_splits = self.args.get("n_splits", 5)

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        train_val_idx, test_idx = next(sss_test.split(torch.arange(num_graphs), labels))

        val_ratio_adjusted = val_ratio / (1 - test_ratio)

        sss_val = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_ratio_adjusted, random_state=42)
        self.splits = []
        for train_idx, val_idx in sss_val.split(train_val_idx, labels[train_val_idx]):
            orig_train_idx = torch.tensor(train_val_idx)[train_idx]
            orig_val_idx = torch.tensor(train_val_idx)[val_idx]
            self.splits.append((orig_train_idx, orig_val_idx, torch.tensor(test_idx)))

        self._set_current_split(0)

    def _set_current_split(self, fold):
        train_idx, val_idx, test_idx = self.splits[fold]
        self.train_data = Subset(self.dataset, train_idx)
        self.val_data = Subset(self.dataset, val_idx)
        self.test_data = Subset(self.dataset, test_idx)
        self.current_fold = fold

    def train_dataloader(self):
        return GraphDataLoader(self.train_data, 
                               batch_size=self.batch_size, 
                               shuffle=True,
                               drop_last=False,
                               )

    def val_dataloader(self):
        return GraphDataLoader(self.val_data, 
                               batch_size=self.batch_size, 
                               shuffle=False,
                               drop_last=False,
                               )

    def test_dataloader(self):
        return GraphDataLoader(self.test_data, 
                               batch_size=self.batch_size, 
                               shuffle=False,
                               drop_last=False,
                               )

    def predict_dataloader(self):
        return GraphDataLoader(self.test_data, 
                               batch_size=self.batch_size, 
                               shuffle=False,
                               drop_last=False,
                               )
    
    def on_train_epoch_end(self):
        next_fold = (self.current_fold + 1) % self.n_splits
        if next_fold == 0:
            self.setup(stage='fit')
        self._set_current_split(next_fold)
    