import pytorch_lightning as pl
from torch_geometric.loader import DataLoader, NeighborLoader
import random
import numpy as np
import torch
from torch_geometric.transforms import RandomNodeSplit

from .createdata import CreateData

class CreateDataModule(pl.LightningDataModule):
    def __init__(self, args, year, seed = 42):
        super().__init__()
        self.args = args
        self.year = year
        self.batch_size = args['batch_size']
        self.num_neighbors = args['num_neighbors']
        self.seed = seed
        self.datasets = {}

    def setup(self, stage=None):
        self.set_seed(self.seed)
        if stage == 'fit' or stage is None:
            self.train_data, self.val_data, _ = self.create_dataloader(self.year)
        if stage == 'test' or stage is None:
            _, _, self.test_data = self.create_dataloader(self.year)

    def create_dataloader(self, year):
        args = self.args
        data = CreateData(args).create_dataset(year)

        batch_size = self.batch_size
        num_neighbors = self.num_neighbors

        transform = RandomNodeSplit(split='train_rest', 
                                    num_val=self.args['val_ratio'],
                                    num_test=self.args['test_ratio'])
        dataset = transform(data)
        self.datasets[year] = dataset 

        train_loader = NeighborLoader(dataset, 
                                      num_neighbors=num_neighbors,
                                      shuffle=True,
                                      directed = False,
                                      input_nodes=dataset.train_mask, 
                                      batch_size=batch_size,
                                      disjoint=False,
                                      is_sorted=True)
        
        val_loader = NeighborLoader(dataset, 
                                    num_neighbors=num_neighbors,
                                    shuffle=False,
                                    directed = False,
                                    input_nodes=dataset.val_mask, 
                                    batch_size=batch_size,
                                    disjoint=False,
                                    is_sorted=True)                      
        
        test_loader = NeighborLoader(dataset, 
                                     num_neighbors=num_neighbors,
                                     shuffle=False,
                                     directed = False,
                                     input_nodes=dataset.test_mask, 
                                     batch_size=batch_size,
                                     disjoint=False,
                                     is_sorted=True)

        return train_loader, val_loader, test_loader

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data

    def test_dataloader(self):
        return self.test_data

    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_node_indices(self, split='train'):
        dataset = self.datasets[self.year]
        node_indices = dataset.node_indices
        return {self.year: node_indices}

    def get_node_ccodes(self, split='train'):
        dataset = self.datasets[self.year]
        node_ccodes = dataset.node_ccodes
        return {self.year: node_ccodes}