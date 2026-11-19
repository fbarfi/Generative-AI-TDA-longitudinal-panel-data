import pandas as pd
from .createdatamodule import CreateDataModule

class AugmentingPanelData():
    def __init__(self, args):

        super(AugmentingPanelData, self).__init__()
        self.args = args
        self.first_year = args['first_year']
        self.last_year = args['last_year']
        self.h_dim = args['h_dim']
        self.n_epochs = args['n_epochs']
        self.latent_size = args['latent_size']
        self.folder = args['folder']
        self.data_prepared = args['data_prepared']
        self.data_homology = args['data_homology']

    def get_data(self,year):
        datamodule = CreateDataModule(self.args,year)
        datamodule.setup()
        dataloaders = {
            "train": datamodule.train_dataloader(),
            "val": datamodule.val_dataloader(),
            "test": datamodule.test_dataloader(),
        }
        return dataloaders

    def masking_data(self, data, mask):
        # Convert boolean mask to indices if needed
        if mask.dtype == torch.bool:
            subset = mask.nonzero(as_tuple=False).view(-1)
        else:
            subset = mask

        edge_index_masked, edge_attr_masked = subgraph(
                                                        subset=subset,
                                                        edge_index=data.edge_index,
                                                        edge_attr=data.edge_attr,
                                                        relabel_nodes=True,
                                                        num_nodes=data.num_nodes 
                                                    )
        return edge_index_masked, edge_attr_masked

    def augmenting_data(self):

        h_dim = self.h_dim
        latent_size = self.latent_size


        path_name_1 = self.data_homology + f'Homology_categorized/{h_dim}/nodes.csv'
        path_name_2 = self.data_homology + f'Homology_categorized/{h_dim}/edges.csv'

        nodes_full = pd.read_csv(path_name_1)
        nodes_full = nodes_full.rename({'graph_id': 'year'}, axis=1)
        edges_full = pd.read_csv(path_name_2).rename({'graph_id': 'year'}, axis=1)

        keys = ['train','val','test']

        csv_dict = {}
        n_years = self.args['nyears']
        for key in keys:

            edges_list = []
            for year in n_years:
                pathfilez = self.folder + f'latentspace/latentspace_csv/'
                pathfilez += f'ARGVA_{key}_latent_features_{year}.csv'
                z = pd.read_csv(pathfilez)

                nodes =nodes_full.loc[nodes_full['year']==year,:].reset_index(drop=True)

                XX_df = nodes.merge(z, how='inner', on=['year']).reset_index(drop=True)
                X_df = XX_df[['year', 'node_id', 'power', 'LatentFeature_0',
                                    'LatentFeature_1']]
                X_df = X_df.sort_values(by=['year', 'node_id']).reset_index(drop=True)
                X_df = X_df.drop_duplicates(subset=['node_id']).reset_index(drop=True)

                edges = edges_full.loc[edges_full['year'] == year, :]

                edges_src = edges['src_id'].values.tolist()
                edges_dst = edges['dst_id'].values.tolist()

                node_src_list = []
                for src in edges_src:
                    node_src_list.append(X_df.loc[X_df['node_id'] == src, :])

                src_df = pd.concat(node_src_list, axis=0).reset_index(drop=True).rename({
                    'node_id': 'src_id',
                    'power': 'power_src',
                    'LatentFeature_0': 'LatentFeature_src_0',
                    'LatentFeature_1': 'LatentFeature_src_1',
                }, axis=1)

                node_dst_list = []
                for dst in edges_dst:
                    node_dst_list.append(X_df.loc[X_df['node_id'] == dst, :])

                dst_df = pd.concat(node_dst_list, axis=0).reset_index(drop=True).rename({
                    'node_id': 'dst_id',
                    'power': 'power_dst',
                    'LatentFeature_0': 'LatentFeature_dst_0',
                    'LatentFeature_1': 'LatentFeature_dst_1',
                }, axis=1)

                dyads_df = pd.concat([src_df[['year', 'src_id']], dst_df['dst_id'],
                                      src_df['power_src'], dst_df['power_dst'],
                                      src_df['LatentFeature_src_0'],
                                      dst_df['LatentFeature_dst_0'],
                                      src_df['LatentFeature_src_1'],
                                      dst_df['LatentFeature_dst_1']], axis=1)

                to_merge_df = pd.concat([dyads_df[['year', 'src_id', 'dst_id']],
                                         dyads_df['power_src'] + dyads_df['power_dst'],
                                         0.5 * (dyads_df['LatentFeature_src_0'] +
                                                dyads_df['LatentFeature_dst_0']),
                                         0.5 * (dyads_df['LatentFeature_src_1'] +
                                                dyads_df['LatentFeature_dst_1'])],
                                        axis=1).rename({
                                             0: 'power',
                                             1: 'LatentFeature_0',
                                             2: 'LatentFeature_1'
                                         }, axis=1)
                to_merge_df.loc[to_merge_df['power'] == 2, 'power'] = 1

                edges_aug = to_merge_df.merge(edges, how='inner', on=['year', 'src_id', 'dst_id'])
                """ assert that we are not dropping any dyads """
                assert edges_aug['src_id'].isin(edges_src).all()
                assert edges_aug['dst_id'].isin(edges_dst).all()

                edges_list.append(edges_aug)

            aug_dyadic_data = pd.concat(edges_list, axis=0).reset_index(drop=True)
            aug_dyadic_data = aug_dyadic_data.rename({
                'power': 'majpower',
                'LatentFeature_0': 'L0',
                'LatentFeature_1': 'L1'
            }, axis=1)
            csv_dict[key] = aug_dyadic_data

        all_dyadic_data = pd.concat([pd.DataFrame(csv_dict[key]) for key in keys], axis=0)
        dd = all_dyadic_data.drop_duplicates(subset=['year','src_id','dst_id'],keep = 'first')

        dd.to_csv(self.folder + f'augmented_longutidinal_panel_data.csv')
        dd.to_csv(self.data_prepared + f'augmented_longutidinal_panel_data.csv')
        print('Done saving augmented data to ARGVA Results and data_prepared folders')

        return