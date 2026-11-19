import pytorch_lightning as pl
from dgl.dataloading import GraphDataLoader
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from collections import Counter

from .dglgraphsdataset import DGLGraphsDatasetClass

class GraphsDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args['batch_size']
        self.val_ratio = args['val_ratio']
        self.n_splits = args.get('n_splits', 5)
        self.train_dataset = None
        self.val_dataset = None
        self.dataset = None
        self.current_fold = 0
        self.splits = None

    def setup(self, stage=None):
        self.dataset = DGLGraphsDatasetClass(self.args)
        
        labels = self.dataset.labels
        num_graphs = len(self.dataset.graphs)

        unique_labels, counts = torch.unique(labels, return_counts=True)
        self.label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.val_ratio, 
                                     random_state=42)
        self.splits = list(sss.split(torch.arange(num_graphs), labels))
        
        self._set_current_split(0)

    def _set_current_split(self, fold):
        train_idx, val_idx = self.splits[fold]
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.current_fold = fold

    def train_dataloader(self):
        return GraphDataLoader(self.train_dataset, 
                               batch_size=self.batch_size, 
                               shuffle=True,  
                               drop_last=False)

    def val_dataloader(self):
        return GraphDataLoader(self.val_dataset, 
                               batch_size=self.batch_size, 
                               shuffle=False, 
                               drop_last=False)

    def predict_dataloader(self):
        return GraphDataLoader(self.val_dataset, 
                               batch_size=self.batch_size, 
                               shuffle=False, 
                               drop_last=False)

    def on_train_epoch_end(self):
        next_fold = (self.current_fold + 1) % self.n_splits
        if next_fold == 0:
            self.setup(stage='fit')
        self._set_current_split(next_fold)
    
