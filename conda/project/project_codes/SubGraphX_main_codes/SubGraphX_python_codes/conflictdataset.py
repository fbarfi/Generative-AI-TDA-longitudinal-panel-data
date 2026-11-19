import pandas as pd
import numpy as np
import torch
import dgl

class ConflictDataset:
    
    def __init__(self, args, graph_id):
        self.args = args
        self.h_dim = args['h_dim']
        self.graph_id = graph_id
        self.Homology = args['Homology']
        self.folder = self.Homology + f'Homology_categorized/{self.h_dim}/'
    
    def UploadData(self):
        folder = self.folder
        graph_id = self.graph_id
        
        all_edges = pd.read_csv(folder + 'edges.csv')
        all_nodes = pd.read_csv(folder + 'nodes.csv')
        
        edges = all_edges[all_edges['graph_id'] == graph_id].drop('graph_id', 
                                            axis=1).reset_index(drop=True)
        nodes = all_nodes[all_nodes['graph_id'] == graph_id].drop('graph_id', 
                            axis=1).reset_index(drop=True).rename({'node_id': 'id'}, axis=1)
        
        return edges, nodes

    def process(self):
        edges_data, nodes_data = self.UploadData()
        
        iso_codes = set(edges_data['src_id']).union(set(edges_data['dst_id']))
        iso_code_to_id = {code: i for i, code in enumerate(iso_codes)}
        
        nodes_data['node_id'] = nodes_data['id'].map(iso_code_to_id)
        nodes_data = nodes_data.sort_values(by=['node_id']).astype(int)
        
        node_features = torch.from_numpy(nodes_data["power"].to_numpy()).float().view(-1, 1)
        node_labels = torch.tensor(nodes_data["disp_weight"].to_numpy()).long()
        
        edge_features = torch.from_numpy(edges_data[['disp','allies', 'contgkf', 'hegdef', 
                        'smldep', 'lcaprat', 'lgmidist', 'smldmat', 'weight']].to_numpy()).float()
        
        edges_src = torch.from_numpy(edges_data["src_id"].map(iso_code_to_id).to_numpy()).long()
        edges_dst = torch.from_numpy(edges_data["dst_id"].map(iso_code_to_id).to_numpy()).long()
        
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['efeat'] = edge_features
        self.graph.edge_ccodes = [torch.tensor(edges_data['src_id']).long(),
                             torch.tensor(edges_data['dst_id']).long()]
        self.graph.year = torch.tensor(self.graph_id).long()
                             
    
    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1