import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class CreateData():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.first_year = args['first_year']
        self.last_year = args['last_year']
        self.h_dim = args['h_dim']
        self.nyears = args['nyears']
        self.data_homology = args['data_homology']
        self.data_original = args['data_original']
        self.minmax_scaler = MinMaxScaler()

    def UploadData(self):
        h_dim = self.h_dim
        edges = pd.read_csv(self.data_homology + f'Homology_categorized/{h_dim}/edges.csv')
        nodes = pd.read_csv(self.data_homology + f'Homology_categorized/{h_dim}/nodes.csv')
        graphs = pd.read_csv(self.data_homology + f'Homology_categorized/{h_dim}/graphs.csv')
        return edges, nodes, graphs

    def extract_ccodes(self,year):
        args = self.args
       
        countries_codes = pd.read_csv(self.data_original+'COW_country_codes.csv')[['ccode', 'statename',
                                                                     'stateabb']]
        _, nodes, _ = self.UploadData()
        ccode_list = nodes.loc[nodes['graph_id'] == year, :].rename({'node_id': 'ccode'}, 
                                                        axis=1)['ccode'].values.tolist()
        ar_ccode = countries_codes.loc[countries_codes.ccode.isin(ccode_list), :].drop_duplicates(
            subset=['ccode'])['ccode'].values
        return ar_ccode

    def convert_ccode_to_name(self, ccode):
        countries = pd.read_csv(self.data_original + 'COW_country_codes.csv')[['ccode', 'stateabb']]
        name = countries.loc[countries['ccode'] == ccode, :]['stateabb'].values[0]
        return name

    def create_data(self,year):
        args = self.args
        first_year = self.first_year
        last_year = self.last_year

        all_edges, all_nodes, all_labels = self.UploadData()

        assert(year in range(first_year, last_year + 1))

        edges = all_edges.loc[all_edges['graph_id'] == year, :].drop(
            ['graph_id'], axis=1).reset_index(drop=True).rename(
            {'src_id': 'i', 'dst_id': 'j'}, axis=1)

        y_df = edges[['i', 'j', 'disp']]
        edges = edges.drop(['disp'], axis=1)

        nodes = all_nodes.loc[all_nodes['graph_id'] == year, :].drop(
            ['graph_id'], axis=1).reset_index(drop=True).rename({'node_id': 'iso_code'}, 
                                                                axis=1)

        labels = all_labels.loc[all_labels['graph_id'] == year, :].drop(
            ['graph_id'], axis=1).values

        iso_codes = set(edges['i']).union(set(edges['j']))
        iso_code_to_id = {code: i for i, code in enumerate(iso_codes)}

        edges['i_id'] = edges['i'].map(iso_code_to_id)
        edges['j_id'] = edges['j'].map(iso_code_to_id)

        edge_index = torch.from_numpy(edges[['i_id', 'j_id']].to_numpy(np.int32)).t().to(torch.long)
        edge_attr = torch.from_numpy(edges[args['edges_features']].to_numpy(np.float32))

        yi = y_df.drop(['j'], axis=1).groupby('i', as_index=False).max().rename({'i': 'node_id'},
                                                                axis=1).reset_index(drop=True)
        yj = y_df.drop(['i'], axis=1).groupby('j', as_index=False).max().rename({'j': 'node_id'}, 
                                                                axis=1).reset_index(drop=True)
        yij = pd.concat([yi, yj], axis=0).reset_index(drop=True).groupby('node_id', 
                                                as_index=False).max().reset_index(drop=True)
        yij['id'] = yij['node_id'].map(iso_code_to_id)
        yy = yij.sort_values('id')['disp'].values
        y = torch.from_numpy(yy).long()

        nodes['id'] = nodes['iso_code'].map(iso_code_to_id)
        nodes = nodes.sort_values('id')['power']

        yi = y_df.drop(['j'], axis=1).groupby('i', as_index=False).sum().rename({'i': 'node_id'}, 
                                                                axis=1).reset_index(drop=True)
        yj = y_df.drop(['i'], axis=1).groupby('j', as_index=False).sum().rename({'j': 'node_id'}, 
                                                                axis=1).reset_index(drop=True)
        yij = pd.concat([yi, yj], axis=0).reset_index(drop=True).groupby('node_id', 
                                                    as_index=False).max().reset_index(drop=True)
        yij['id'] = yij['node_id'].map(iso_code_to_id)
        yy = yij.sort_values('id')['disp'].values
        disp_weight = pd.DataFrame({'disp_weight':
                            self.minmax_scaler.fit_transform(yy.reshape((-1, 1))).reshape((-1,))})

        nodes = pd.concat([disp_weight, nodes], axis=1)
        x = torch.from_numpy(nodes.to_numpy(np.float32))

        return x, edge_index, edge_attr, y, yy

    def create_dataset(self,year):
        
        x, edge_index, edge_attr, y, yy = self.create_data(year)
        dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        _, _, graphs = self.UploadData()
        labels = graphs.loc[graphs['graph_id'] == year, 'label'].to_numpy(dtype=np.float32)
        labels_tensor = torch.from_numpy(labels+1)
        
        dataset.glabel  = labels_tensor.long() 
        
        dataset.num_nodes = dataset.x.shape[0]
        ar_ccodes = self.extract_ccodes(year)
        dataset.node_ccodes = torch.from_numpy(ar_ccodes).long()
     
        dataset.num_classes = np.unique(yy).shape[0]
        dataset.num_edges = dataset.edge_index.shape[1]
        dataset.year = torch.full((dataset.num_nodes,), year, dtype=torch.long)

        dataset.node_indices = torch.arange(dataset.num_nodes).numpy()
        
        return dataset