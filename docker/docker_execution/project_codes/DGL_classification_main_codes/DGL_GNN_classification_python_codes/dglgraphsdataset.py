import pandas as pd
import torch
import dgl
from dgl.data import DGLDataset

class DGLGraphsDataset(DGLDataset):
    
    def __init__(self, args):
       
        self.args = args
        self.h_dim = args['h_dim']
        self.data_homology = args['data_homology']
        self.folder = self.data_homology + f'Homology_categorized/{self.h_dim}/'
        self.graphs = []
        self.labels = []
        self.all_edges = pd.read_csv(self.folder + 'edges.csv')
        self.all_nodes = pd.read_csv(self.folder + 'nodes.csv')
        self.all_graphs = pd.read_csv(self.folder + 'graphs.csv')
        super(DGLGraphsDataset, self).__init__('DGLGraphsDatasetClass')
    
    def process(self):
        folder = self.folder

        graph_ids = self.all_graphs['graph_id'].unique()
        
        for graph_id in graph_ids:
           
            edges_data = self.all_edges[self.all_edges['graph_id'] == graph_id].drop('graph_id', 
                                                            axis=1).reset_index(drop=True)
            nodes_data = self.all_nodes[self.all_nodes['graph_id'] == graph_id].drop('graph_id', 
                                                            axis=1).reset_index(drop=True)

            # Map ISO codes to unique node IDs
            iso_codes = pd.concat([edges_data['src_id'], edges_data['dst_id']]).unique()
            iso_code_to_id = {code: i for i, code in enumerate(iso_codes)}

            # Map node IDs for nodes_data
            nodes_data['node_ccode'] = nodes_data['node_id']
            nodes_data['node_id'] = nodes_data['node_id'].map(iso_code_to_id)
            nodes_data.sort_values(by=['node_id'], inplace=True)
            
            # Extract node and edge features efficiently
            node_features = torch.tensor(nodes_data[["power", "disp_weight"]].values, dtype=torch.float32)
            edge_features = torch.tensor(edges_data[['allies', 'contgkf', 'smldep', 'lcaprat', 
                                        'lgmidist', 'smldmat', 'weight']].values, dtype=torch.float32)
            
            # Map source and destination node IDs
            edges_src = torch.tensor(edges_data['src_id'].map(iso_code_to_id).values, dtype=torch.int64)
            edges_dst = torch.tensor(edges_data['dst_id'].map(iso_code_to_id).values, dtype=torch.int64)
            
            graph = dgl.graph((edges_src, edges_dst), num_nodes=len(nodes_data))
            
            # Add node and edge features
            graph.ndata['feat'] = node_features
            graph.ndata['node_ccode'] = torch.tensor(nodes_data['node_ccode'].values, dtype=torch.int64)
            
            graph.edata['efeat'] = edge_features
            graph.edata['edge_ccode'] = torch.stack([
                                torch.tensor(edges_data['src_id'].values, dtype=torch.int64),
                                torch.tensor(edges_data['dst_id'].values, dtype=torch.int64)
                            ], dim=1)
            
            graph.year = torch.tensor(graph_id, dtype=torch.int64)
            
            self.graphs.append(graph)

        # Store labels as a tensor
        self.labels = torch.tensor(self.all_graphs['label'].values, dtype=torch.int64)
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)

