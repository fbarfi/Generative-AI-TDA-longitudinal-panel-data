import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx

class CreateGraphsData():
    def __init__(self, args):
        self.args = args
        self.num_cat = args['num_cat']
        self.file_vars = args['file_vars']
        self.data_original = args['folder']
        self.folder_prep = args['folder_prep']
        self.Homology = args['Homology']
        self.H_list = args['H_list']
        self.n_years = args['n_years']

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_edges_data(self, year):

        data = pd.read_csv(self.folder_prep + f'states_DV&IDVs_pol_igo/states_DV&IDVs_pol_igo_{year}.csv')
        X = data.drop(['disp_lag'], axis=1).reset_index(drop=True)

        edges = X.rename(columns={'state_a': 'src_id', 'state_b': 'dst_id'}).reset_index(drop=True)
        IOmb = pd.read_csv(
            self.folder_prep + f'states_IOmembs_pol_igo/states_IOmembs_pol_igo_{year}.csv',
            usecols=['row_io_sum']).rename(columns={'row_io_sum': 'weight'}).reset_index(drop=True)

        edges = pd.concat([IOmb, edges.iloc[:, 1:]], axis=1)
        return edges

    def preprocess_edges(self, edges):

        edges_float = edges[['smldep', 'lcaprat', 'lgmidist', 'smldmat', 'weight']]
        edges_float = pd.DataFrame(data=self.scaler.fit_transform(edges_float.values),
                                   columns=edges_float.columns)
        edges = pd.concat([edges[['year', 'src_id', 'dst_id', 'disp', 'allies', 'contgkf', 'hegdef']],
                           edges_float], axis=1).reset_index(drop=True).astype('float32')
        for v in ['src_id', 'dst_id']:
            edges[v] = edges[v].astype(np.int32)
        return edges.rename({'year': 'graph_id'}, axis=1)

    def process_nodes(self, year, edges):

        G = nx.Graph()
        G.add_edges_from(edges[['src_id', 'dst_id']].values.tolist())
        nodes = sorted(G.nodes())

        majpowers = pd.read_csv(self.data_original + 'majors2016.csv',
                                usecols=['ccode']).ccode.values.tolist()
        x = pd.DataFrame({'power': np.zeros(len(nodes))}, index=nodes)
        x.loc[x.index.isin(majpowers), 'power'] = 1
        nodes_types = x.reset_index().rename({'index': 'node_id'}, axis=1)

        disp_sums_src = edges.groupby('src_id')['disp'].sum().reset_index().rename(columns={'src_id':
                                                                    'node_id', 'disp': 'disp_weight'})
        disp_sums_dst = edges.groupby('dst_id')['disp'].sum().reset_index().rename(columns={'dst_id':
                                                                    'node_id', 'disp': 'disp_weight'})

        disp_sums = pd.concat([disp_sums_src,
                        disp_sums_dst]).groupby('node_id')['disp_weight'].sum().reset_index()

        nodes_types = nodes_types.merge(disp_sums, how='left', on='node_id')

        nodes_types['disp_weight'] = nodes_types['disp_weight'].fillna(0)

        return pd.concat([pd.DataFrame({'graph_id': np.tile(year,
                                        len(nodes_types))}).astype('int'), nodes_types], axis=1)

    def loading_data(self, year):
        edges = self.load_edges_data(year)
        return self.preprocess_edges(edges)

    def homology_values(self):
        H_list = self.H_list
        n_years = self.n_years

        hom_list = []
        for year in n_years:
            file = self.Homology + f'IO_data_hom/IO_data_hom_H_012_{year}.csv'
            hom_df = pd.read_csv(file, usecols=H_list).drop_duplicates(keep='first',
                                                                       ignore_index=True)
            hom_list.append(hom_df)

        df_H = pd.concat([*hom_list], axis=0).reset_index(drop = True)

        targetsDict = {}
        for H in H_list:
            df = (df_H[H] - df_H[H].min()) / (df_H[H].max() - df_H[H].min())

            a, b = df.min(), df.max()
            bin_edges = np.linspace(a, b, self.num_cat)
            DD = pd.cut(df.values, right=True, include_lowest=True, bins=bin_edges,
                        labels=[f'{H}_Cat_{k}' for k in range(self.num_cat - 1)])

            y_df = pd.DataFrame({'norm_hom': DD}).astype('category')
            y_df = pd.DataFrame(data=self.label_encoder.fit_transform(y_df['norm_hom'].values),
                                columns=['label'])

            targetsDict[H] = pd.concat([pd.DataFrame({'graph_id': n_years}), y_df], axis=1)


        return targetsDict

    def CreateData(self, year):
        edges = self.loading_data(year)
        edges_list = edges[['graph_id', 'src_id', 'dst_id']].astype('int')
        edges_features = edges[['disp', 'allies', 'contgkf', 'hegdef', 'smldep',
                                'lcaprat', 'lgmidist', 'smldmat', 'weight']]

        nodes = self.process_nodes(year, edges)

        return edges_list, edges_features, nodes

    def generate_data(self):
        for graph_id in self.n_years:
            yield self.CreateData(graph_id)

    def extract_ccodes(self, year):
        countries_codes = pd.read_csv(self.data_original + 'COW_country_codes.csv')[['ccode',
                                                                        'statename', 'stateabb']]

        _, _, nodes = self.CreateData(year)

        ccode_list = nodes['node_id'].values.tolist()
        ar_ccode = countries_codes[countries_codes['ccode'].isin(ccode_list)]['ccode'].values

        return ar_ccode

    def CreateDataset(self, H):

        e_list, e_features, n_list = zip(*self.generate_data())
        edges = pd.concat(e_list, axis=0)
        edges_features = pd.concat(e_features, axis=0)
        edges = pd.concat([edges, edges_features], axis=1)
        nodes = pd.concat(n_list, axis=0).astype('int')

        homDict = self.homology_values()

        graphs = homDict[H].astype('int')

        return edges, nodes, graphs

    def create_pyg_dataset(self, H):

        data_dir = self.Homology + f'Homology_categorized/{H}'
        edge_file = os.path.join(data_dir, 'edges.csv')
        node_file = os.path.join(data_dir, 'nodes.csv')
        graph_file = os.path.join(data_dir, 'graphs.csv')

        edges = pd.read_csv(edge_file)
        nodes = pd.read_csv(node_file)
        graphs = pd.read_csv(graph_file)

        data_list = []

        for graph_id in graphs['graph_id'].unique().tolist():

            graph = graphs.loc[graphs['graph_id'] == graph_id, :].reset_index(drop=True)

            edge_index = edges[edges['graph_id'] == graph_id][['src_id', 'dst_id']].values.T
            edge_attr = torch.tensor(edges[edges['graph_id'] == graph_id].drop(['graph_id', 'src_id',
                                                                        'dst_id'], axis=1).values,
                                     dtype=torch.float)

            node_mask = nodes['graph_id'] == graph_id
            node_features = torch.tensor(nodes[node_mask].drop(['graph_id', 'node_id'],
                                                               axis=1).values, dtype=torch.float)

            y = torch.tensor(graph['label'].values, dtype=torch.long)

            dataset = Data(x=node_features,
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_attr=edge_attr,
                        y=y)

            labels = graph['label'].values
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            dataset.glabel = labels_tensor
            dataset.num_nodes = dataset.x.shape[0]

            ar_ccodes = self.extract_ccodes(graph_id)
            dataset.node_ccodes = torch.from_numpy(ar_ccodes).long()

            dataset.num_classes = np.unique(labels).shape[0]
            dataset.num_edges = dataset.edge_index.shape[1]
            dataset.year = torch.full((dataset.num_nodes,), graph_id, dtype=torch.long)

            data_list.append(dataset)

        return data_list

    def save_to_csv(self, data, file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(file_path, index=False)

    def save_graphs_data(self):

        for H in self.H_list:
            edges, nodes, graphs = self.CreateDataset(H)
            data_dir = self.Homology + f'Homology_categorized/{H}'

            self.save_to_csv(edges, os.path.join(data_dir, 'edges.csv'))
            self.save_to_csv(nodes, os.path.join(data_dir, 'nodes.csv'))
            self.save_to_csv(graphs, os.path.join(data_dir, 'graphs.csv'))

            dataset = self.create_pyg_dataset(H)

        return dataset
