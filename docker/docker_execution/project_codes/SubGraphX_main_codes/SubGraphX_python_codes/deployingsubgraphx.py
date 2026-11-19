import torch
from tqdm import tqdm
import dgl
from dgl.data.utils import save_graphs, load_graphs
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from .dglgraphsdataset import DGLGraphsDatasetClass

class DeploySubGraphXClass:
    def __init__(self, args):
        super(DeploySubGraphXClass, self).__init__()

        self.args = args
        self.first_year = args['first_year']
        self.last_year = args['last_year']
        self.h_dim = args['h_dim']
        self.n_years = args['n_years']

        self.folder_results = args['folder_results'] 
        self.data_original = args['data_original']
        self.Homology = args['Homology']
        self.graphs_data = self.Homology + f'Homology_categorized/{self.h_dim}/'
        self.ccodes_file = self.data_original + 'COW_country_codes.csv'
        self.majors = self.data_original + 'majors2016.csv'
        self.nodes_features = ['graph_id', 'node_id','disp_weight'] 
        self.edges_features = ['disp','allies', 'contgkf', 'hegdef', 'smldep', 
                               'lcaprat', 'lgmidist', 'smldmat', 'weight']

        self.countries_codes = pd.read_csv(self.ccodes_file)
        self.majpowers = pd.read_csv(self.majors)['ccode'].unique().tolist()
        self.nodes = pd.read_csv(self.graphs_data + 'nodes.csv')
        self.countries = pd.read_csv(self.ccodes_file, usecols =['ccode', 'stateabb'])
        self.dataset = DGLGraphsDatasetClass(self.args).graphs
        

    def convert_ccode_to_name(self, ccode):
        ccodes = self.countries_ccodes['ccode'].tolist()
        name = self.countries_ccodes['stateabb'].tolist()[ccodes.index(ccode)]
        return name

    def ReconstructingEdgesDatafromSubgraph(self, year):
        h_dim = self.h_dim
        edges_features = self.edges_features
        nodes_data =  self.nodes.loc[self.nodes['graph_id'] ==year,:][['node_id',
                            'disp_weight']].rename({'node_id':'ccode'}, axis =1).reset_index(drop = True)
      
        states_file = self.folder_results + f'SubGraphX/{h_dim}_nodes_info_{year}.csv' 
        edges_file = self.folder_results + f'SubGraphX/{h_dim}_edges_info_{year}.csv'
       
        subgraph_edges_df = pd.read_csv(edges_file)
        
        states_names = pd.read_csv(states_file)['node_name']
        states_ccodes =  pd.read_csv(states_file)['node_ccode']

        graph_file = self.folder_results+ f'SubGraphX/{h_dim}_subgraphs_{year}.bin'
        G = load_graphs(graph_file)[0][0]

        subgraph_edges_df['src_ccode'] = subgraph_edges_df['dyadid'] //1000
        subgraph_edges_df['dst_ccode'] = subgraph_edges_df['dyadid'] % 1000
        subgraph_edges_df['majpower'] = 0
        subgraph_edges_df.loc[subgraph_edges_df['src_ccode'].isin(self.majpowers), 'majpower'] = 1
        subgraph_edges_df.loc[subgraph_edges_df['dst_ccode'].isin(self.majpowers), 'majpower'] = 1
        
        edges_features_df = pd.DataFrame(
            data=G.edata['efeat'].detach().numpy(), 
            columns=edges_features
        )
        
        subgraph_df = pd.concat([subgraph_edges_df, edges_features_df], axis=1)
        
        return subgraph_df

    def ReConstructDatafromSubgraph(self):
        h_dim = self.h_dim
        n_years = self.n_years
        
        subgraphs = []
        for year in tqdm(n_years):
            subgraph_df = self.ReconstructingEdgesDatafromSubgraph(year)

            subgraph_file = self.folder_results+f'Reconstructed_data/{h_dim}_SubGraphX_important_data_{year}.csv'
            subgraph_df.to_csv(subgraph_file, index=None)

            subgraph_df = pd.concat([pd.DataFrame({'year': np.tile(year, 
                                                        len(subgraph_df))}), subgraph_df], axis=1)
            subgraphs.append(subgraph_df)

        subgraphs_df = pd.concat(subgraphs, axis=0).reset_index(drop=True)

        file_all = f'{self.folder_results}Reconstructed_data/{h_dim}_SubGraphX_important_data_all.csv'
        subgraphs_df.to_csv(file_all, index=None)
        print('Saved all graphs in one file')

        return
