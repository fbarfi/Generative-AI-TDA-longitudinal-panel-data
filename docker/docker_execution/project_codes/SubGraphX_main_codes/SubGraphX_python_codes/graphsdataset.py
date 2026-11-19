from dgl.data import DGLDataset
import pandas as pd
import numpy as np
import torch

from .conflictdataset import ConflictDataset

class GraphsDataset(DGLDataset):
    
    def __init__(self, args):
        self.args = args
        self.h_dim = args['h_dim']
        self.graph = ConflictDataset
        self.Homology = args['Homology']
        self.folder = self.Homology + f'Homology_categorized/{self.h_dim}/'
        super(GraphsDataset, self).__init__(name="multi_dataset")
    
    def UploadData(self):
        folder = self.folder
        
        all_edges = pd.read_csv(folder + 'edges.csv')
        all_nodes = pd.read_csv(folder + 'nodes.csv')
        all_graphs = pd.read_csv(folder + 'graphs.csv')
       
        return all_edges, all_nodes, all_graphs
         
    def process(self):
        all_edges, all_nodes, all_graphs = self.UploadData()
    
        self.graphs = []
        self.labels = []
        
        graph_ids = all_graphs['graph_id'].unique()
        
        for graph_id in graph_ids:
            conflict_graph = self.graph(self.args, graph_id)
            conflict_graph.process()  
            
            self.graphs.append(conflict_graph.graph)
            label = all_graphs.loc[all_graphs['graph_id'] == graph_id, 'label'].values[0]
            self.labels.append(label)
        
        self.labels = torch.tensor(self.labels).long()
        self.gclasses = len(torch.unique(self.labels))
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)

