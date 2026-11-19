from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize,NoNorm
import matplotlib
import matplotlib.patches as patches

import networkx as nx
import numpy as np
import pandas as pd


class PlottingSubgraphXClass():

    def __init__(self,args):

        super(PlottingSubgraphXClass,self).__init__()

        self.args = args
        self.first_year = args['first_year']
        self.last_year = args['last_year']
        self.years = args['specific_years']
        self.h_dim = args['h_dim']
        self.folder = args['folder_results'] + 'SubGraphX/'
        self.Homology = args['Homology']
        self.data_original = args['data_original']
        self.folder_results = args['folder_results']
        self.countries_codes = pd.read_csv(self.data_original+'COW_country_codes.csv').rename({'ccode':'node_id'},
                                                                                              axis=1)
        self.nodes_df = pd.read_csv(self.Homology +f'/Homology_categorized/{self.h_dim}/nodes.csv')

    def year_indices(self,year):
        years = self.years
        year_idx = {year:i for i,year in enumerate(years)}
        return year_idx[year]

    def upload_subgraph(self,year):

        folder = self.folder
        graph_file = folder + f'H_1_subgraphs_{year}.bin'

        subgraph,_ = load_graphs(graph_file)

        return subgraph[0]

    def converting_node_ids_to_statenames(self,g_nodes,year):

        nodes_df = self.nodes_df
        unique_nodes_df = nodes_df['node_id'].drop_duplicates().sort_values().reset_index(drop = True)

        df = self.countries_codes.loc[self.countries_codes.node_id.isin(unique_nodes_df.values),
                            :].drop_duplicates(subset =['node_id']).reset_index(drop = True)
        D = nodes_df.loc[nodes_df['graph_id']==year,:].reset_index(drop=True)
        D = D.reset_index().rename({'index':'node_index'},axis=1)
        D_merged = D.merge(df, how = 'left', on = ['node_id'])[['graph_id','node_index',
                                                                'node_id','stateabb']]
        g_states =  D_merged.loc[D_merged['node_index'].isin(g_nodes),:]['stateabb'].values.tolist()

        return g_states

    def plot_subgraph(self):

        years = self.years

        cmap = matplotlib.colormaps.get_cmap('seismic')
        fig,axes = plt.subplots(2,2,figsize = (14,14))
        axes = axes.flatten()

        for year, ax in zip(years,axes):

            g = self.upload_subgraph(year)
            n_states = g.ndata['_ID'].detach().numpy().tolist()
            g_states = self.converting_node_ids_to_statenames(n_states,year)
            g_nodes = g.nodes().detach().numpy().tolist()

            node_weights = g.ndata['feat'].detach().numpy()[:,-1]
            nodes_labels =  nodes_labels = {i: f'{lab}' for i,lab in enumerate(g_states)}


            vmin,vmax = min(node_weights), max(node_weights)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            node_colors = [cmap(norm(weight)) for weight in node_weights]
            G_nx = g.to_networkx()
            pos = nx.nx_agraph.graphviz_layout(G_nx)

            nx.draw(G_nx, pos, ax=ax, with_labels = False, node_size = 600,
                                   node_color=node_colors)
            nx.draw_networkx_labels(G_nx, pos, ax=ax, labels=nodes_labels, font_size=10,
                                font_color="white", font_family='sans-serif')

            edge_weights = g.edata['efeat'].numpy()[:,-1]
            edges = g.edges()
            edges_dict = {}
            for i, (a,b) in enumerate(zip(edges[0],edges[1])):
                edges_dict[(a.item(),b.item())]= np.round(edge_weights[i],decimals = 2)

            cax = nx.draw_networkx_edge_labels(G_nx, pos, ax=ax, edge_labels=edges_dict,
                                         font_color='black', font_size=10)
            ax.margins(.15)
            ax.set_title(f'{year}',fontsize = 18,weight = 'bold')

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            border = patches.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                                       fill=False, edgecolor='black', linewidth=5)
            ax.add_patch(border)

            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # You can set an empty array
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Disputes', fontsize=14, weight = 'bold')
            cbar.set_ticklabels([])

        plt.suptitle('Subgraphs for Multiple Years',fontsize = 25,weight = 'bold')
        plt.tight_layout()
        pathname = self.folder_results + 'SubGraphX/SubgraphX_4_years.png'
        plt.savefig(pathname)
        plt.close()

        return