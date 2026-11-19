import torch.nn as nn
import torch
import dgl
from dgl.nn import SubgraphX
from dgl.data.utils import save_graphs
import pandas as pd
import gc

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx

from .runmodel import RunModel

class SubGraphXExplainer(nn.Module):
    def __init__(self, args):
        super(SubGraphXExplainer, self).__init__()

        self.args = args
        self.first_year = args['first_year']
        self.last_year = args['last_year']
        self.num_hops = args['num_hops']
        self.num_rollouts = args['num_rollouts']
        self.num_child = args['num_child']
        self.node_min = args['node_min']
        self.h_dim = args['h_dim']
        self.folder_results = args['folder_results']
        self.log = args['log']
        self.shapley_steps = args['shapley_steps']
        self.coef = args['coef']
        self.data_original = args['data_original']
        self.device = self.args['device']
        self.run_model = self.upload_model()
       

        self.countries_ccodes = pd.read_csv(self.data_original + 'COW_country_codes.csv')
        self.ccode_to_name = dict(zip(self.countries_ccodes['ccode'],
                                      self.countries_ccodes['stateabb']))

        dgl.seed(43)

    def upload_model(self):
        checkpoint_path = self.folder_results + 'best_model.ckpt'
        model = RunModel.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            model=RunModel(self.args),
            args=self.args
        )
        model.to(self.device)
        return model

    def convert_ccode_to_name(self, ccode):
        return self.ccode_to_name.get(ccode, str(ccode))

    def find_single_subgraph(self, graph, glabel, year):
        edge_gatconv_model = self.run_model.model

        node_features = graph.ndata['feat']
        edge_features = graph.edata['efeat']

        kwargs = {'efeat': edge_features}

        explainer = SubgraphX(
            edge_gatconv_model,
            num_hops=self.num_hops,
            num_rollouts=self.num_rollouts,
            num_child=self.num_child,
            node_min=self.node_min,
            shapley_steps=self.shapley_steps,
            coef=self.coef,
            log=self.log
        )

        g_nodes_explain = explainer.explain_graph(graph=graph,
                                                  feat=node_features,
                                                  target_class=glabel,
                                                  **kwargs)

        g_nodes = g_nodes_explain.detach().numpy().tolist()

        subgraph = dgl.node_subgraph(graph, g_nodes, relabel_nodes=True, store_ids=True)

        save_graphs(self.folder_results + f'SubGraphX/{self.h_dim}_subgraphs_{year}.bin',
                    [subgraph])

        src_edges = subgraph.edges()[0].detach().numpy()
        dst_edges = subgraph.edges()[1].detach().numpy()
        src_edge_ccodes = subgraph.ndata['node_ccode'][src_edges]
        dst_edge_ccodes = subgraph.ndata['node_ccode'][dst_edges]

        edge_dyadids = pd.DataFrame({
            'src_name': [self.convert_ccode_to_name(ccode) for ccode in src_edge_ccodes],
            'dst_name': [self.convert_ccode_to_name(ccode) for ccode in dst_edge_ccodes],
            'dyadid': 1000 * src_edge_ccodes + dst_edge_ccodes,
        })

        filee = self.folder_results + f'SubGraphX/{self.h_dim}_edges_info_{year}.csv'
        edge_dyadids.to_csv(filee, index=False)

        node_ccodes = subgraph.ndata['node_ccode'].detach().numpy()
        node_names = [self.convert_ccode_to_name(ccode) for ccode in node_ccodes]
        g_states = pd.DataFrame({'node_ccode': node_ccodes, 'node_name': node_names})

        filen = self.folder_results + f'SubGraphX/{self.h_dim}_nodes_info_{year}.csv'
        g_states.to_csv(filen, index=False)


        del explainer
        gc.collect()

        return #subgraph, nodes_labels

    def explain(self, graph, glabel, year):

        self.find_single_subgraph(graph, glabel, year)

        return

