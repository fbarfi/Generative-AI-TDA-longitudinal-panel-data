import pytorch_lightning as pl
from dgl.dataloading import GraphDataLoader
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np

from .dglgraphsdataset import DGLGraphsDataset


class GraphsDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args['batch_size']
        self.val_ratio = args['val_ratio']
        self.test_ratio = args['test_ratio']
        self.dataset = None
        self.setup()

    def setup(self, stage=None):
       
        self.dataset = DGLGraphsDataset(self.args)

        labels = self.dataset.labels.numpy() 
        num_graphs = len(self.dataset.graphs)
        
        train_val_idx, test_idx = train_test_split(
            range(num_graphs),
            test_size=self.test_ratio,
            stratify=labels,
            random_state=42
        )
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.val_ratio / (1 - self.test_ratio),
            stratify=labels[train_val_idx],
            random_state=42
        )

        self.train_data = Subset(self.dataset, train_idx)
        self.val_data = Subset(self.dataset, val_idx)
        self.test_data = Subset(self.dataset, test_idx)

    def train_dataloader(self):
        return GraphDataLoader(self.train_data, 
                               batch_size=self.batch_size, 
                               shuffle=False,
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
        return GraphDataLoader(self.val_data, 
                               batch_size=self.batch_size, 
                               shuffle=False,
                               drop_last=False,
                               )

