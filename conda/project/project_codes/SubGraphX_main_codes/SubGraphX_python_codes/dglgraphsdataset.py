import pandas as pd
import torch
import dgl
from dgl.data import DGLDataset

class DGLGraphsDatasetClass(DGLDataset):
    
    def __init__(self, args):
        self.args = args
        self.h_dim = args['h_dim']
        self.Homology = args['Homology'] 
        self.folder = self.Homology + f'Homology_categorized/{self.h_dim}/'
        self. all_edges = pd.read_csv(self.folder + 'edges.csv')
        self.all_nodes = pd.read_csv(self.folder + 'nodes.csv')
        self.graphs = []
        self.labels = []
        super(DGLGraphsDatasetClass, self).__init__('DGLGraphsDatasetClass')

    def convert_ccode_to_name(self, ccode):
        name = self.countries.loc[self.countries['ccode'] == ccode, :]['stateabb'].values[0]
        return name
    
    def UploadData(self, graph_id):
        
        edges = self.all_edges[self.all_edges['graph_id'] == graph_id].drop('graph_id', 
                                                                axis=1).reset_index(drop=True)
        
        nodes = self.all_nodes[self.all_nodes['graph_id'] == graph_id].drop('graph_id', 
                                            axis=1).reset_index(drop=True).rename({'node_id': 'id'}, 
                                                                                  axis=1)
        nodes['node_ccode'] = nodes['id']
        
        return edges, nodes
        
    def process(self):

        folder = self.folder
        all_graphs = pd.read_csv(folder + 'graphs.csv')
        graph_ids = all_graphs['graph_id'].unique()

        for graph_id in graph_ids:
            
            edges_data, nodes_data = self.UploadData(graph_id)
            
            iso_codes = set(edges_data['src_id']).union(set(edges_data['dst_id']))
            iso_code_to_id = {code: i for i, code in enumerate(iso_codes)}
            
            node_features = torch.from_numpy(nodes_data[["power","disp_weight"]].to_numpy()).float()
            # node_labels = torch.tensor(nodes_data["disp_weight"].to_numpy()).long()
            
            edge_features = torch.from_numpy(edges_data[['disp','allies', 'contgkf', 'hegdef', 
                                    'smldep', 'lcaprat', 'lgmidist', 'smldmat', 
                                                         'weight']].to_numpy()).float()
            
            edges_src = torch.from_numpy(edges_data["src_id"].map(iso_code_to_id).to_numpy()).long()
            edges_dst = torch.from_numpy(edges_data["dst_id"].map(iso_code_to_id).to_numpy()).long()
            
            graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
            
            graph.ndata['feat'] = node_features
            graph.ndata['node_ccode'] = torch.tensor(nodes_data['node_ccode'].values,dtype=torch.int64)
            #
            
            graph.edata['efeat'] = edge_features
            graph.edata['edge_ccode'] = torch.stack([
                                torch.tensor(edges_data['src_id'].values, dtype=torch.int64),
                                torch.tensor(edges_data['dst_id'].values, dtype=torch.int64)
                            ], dim=1)
            graph.year = torch.tensor(graph_id).long()
            
            self.graphs.append(graph)

        self.labels =  all_graphs['label'].values.tolist()
        self.labels = torch.LongTensor(self.labels)
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)
