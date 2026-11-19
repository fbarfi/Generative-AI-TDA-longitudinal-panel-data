import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import UnivariateSpline
import networkx as nx
import xarray as xr


from .graphsdatamodule import GraphsDataModule
from .rungnnmodel import RunGNNModel

class AttentionWeightsAnalysis():
    def __init__(self, args, data_module):
        self.args = args
        self.first_year = args['first_year']
        self.last_year = args['last_year']
        self.h_dim = args['h_dim']
        self.model_path = args['folder'] + 'GNN_models/model_path.ckpt'
        self.models_folder = args['models_folder']
        self.data_original = args['data_original']
        self.data_prepared = args['data_prepared']
        self.data_homology = args['data_homology']
        self.graphs = self.data_homology + f'Homology_categorized/{self.h_dim}/'
        self.folder = args['folder']
        self.out_channels = args['out_channels']
        self.phases = ['train','val','test']
        self.data_module = data_module

    
    def convert_ccode_to_name(self, ccode):
        countries = pd.read_csv(self.data_original + 'COW_country_codes.csv')[['ccode', 'stateabb']]
        name = countries.loc[countries['ccode'] == ccode, :]['stateabb'].values[0]
        return name

    def get_data(self):                                      
        data_module = GraphsDataModule(self.args)
        data_module.setup()
        train_data = data_module.train_data
        val_data = data_module.val_data
        test_data = data_module.test_data
        return train_data, val_data, test_data

    def loading_model(self):
        checkpoint_path = self.args['models_folder'] + 'saved_model.ckpt'
        model = RunGNNModel.load_from_checkpoint(
            checkpoint_path = checkpoint_path,
            model=RunGNNModel(self.args),
            args=self.args
        )
        return model


    def assign_attention_weights_to_graphs(self):

        attw = {'train': [], 'val': [], 'test': []}
        years = {'train': [], 'val': [], 'test': []}

        train_loader, val_loader, test_loader = self.get_data()
        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        for phase in self.phases:
            checkpoint_path = f"{self.models_folder}GNN_attention_weights_{phase}_last_epoch.ckpt"
            attws = torch.load(checkpoint_path)

            for att, data in zip(attws, dataloaders[phase]):
                g, _ = data
                aa = att[1]
                aa_sum = aa.sum(dim=0, keepdim=True) 
                aa_normalized = aa / aa_sum
                year_tensor = torch.full((aa.size(0),), g.year.item(), dtype=torch.long)

                attw[phase].append(aa_normalized)
                years[phase].append(year_tensor)


        att_weights = torch.cat([torch.cat(attw[phase], dim=0) for phase in self.phases], dim=0).reshape(-1, 1)
        years_combined = torch.cat([torch.cat(years[phase], dim=0) for phase in self.phases], dim=0)

        att_weights_np = att_weights.detach().numpy()
        years_np = years_combined.detach().numpy()

        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(att_weights_np)

        distances = torch.cdist(torch.tensor(att_weights_np), torch.tensor(kmeans.cluster_centers_))
        attw_probas = torch.exp(-distances)
        attw_probas /= attw_probas.sum(dim=1, keepdim=True)
        attw_probas = attw_probas.numpy()  

        print('Done with assign_attention_weights_to_graphs')

        return attw_probas, years_np, kmeans


    def plot_weights_probabilities(self):
        print("[INFO] Starting to plot attention weight probabilities...")

        probabilities, years, _ = self.assign_attention_weights_to_graphs()
        years = np.array(years).astype(int).flatten()

        unique_years = np.sort(np.unique(years))
        num_clusters = probabilities.shape[1]
        year_probabilities = np.zeros((len(unique_years), num_clusters))

        for i, year in enumerate(unique_years):
            indices = (years == year)
            year_probabilities[i] = np.nanmean(probabilities[indices], axis=0)
       
        cumulative_probabilities = np.cumsum(year_probabilities, axis=1)

        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster_idx in range(num_clusters):
            if cluster_idx == 0:
                ax.fill_between(unique_years, 0, cumulative_probabilities[:, cluster_idx],
                                label=f'Cluster {cluster_idx}')
            else:
                ax.fill_between(unique_years,
                                cumulative_probabilities[:, cluster_idx - 1],
                                cumulative_probabilities[:, cluster_idx],
                                label=f'Cluster {cluster_idx}')

        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Proportions of Topological\nClass Probabilities')
        ax.legend()

        plt.tight_layout()
        save_path = self.folder + 'plots/attention_weights_probabilities_proportions_over_years.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        gc.collect()


    def smooth_with_splines(self, y):
        y = np.array(y)
        x = np.arange(y.shape[0])
        spline = UnivariateSpline(x, y, s=10,k=5)
        y_smooth = spline(x)
        return y_smooth


    def compute_silhouette_scores(self):
        attw_probas, years_combined, kmeans = self.assign_attention_weights_to_graphs()
        silhouette_scores = {}
        labels_np = kmeans.labels_

        for year in np.unique(years_combined):
            indices = (years_combined == year)
            if indices.sum() > 1 and np.unique(labels_np[indices]).size > 1:
                score = silhouette_score(attw_probas[indices], labels_np[indices])
                silhouette_scores[year] = score

        return silhouette_scores
    

    def plot_silhouette_scores(self):
        print("[INFO] Computing silhouette scores...")
        silhouette_scores = self.compute_silhouette_scores()

        years = list(silhouette_scores.keys())
        scores = list(silhouette_scores.values())

        plt.figure(figsize=(10, 6))
        plt.plot(years, scores, marker='o')
        plt.title('Silhouette Scores Over Time')
        plt.xlabel('Year')
        plt.ylabel('Silhouette Score')
        plt.grid(True)

        save_path = self.folder + 'plots/silhouette_scores.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')        
        plt.close()
        gc.collect()


    def find_top_important_edges(self):
        top_k = 8
        model = self.loading_model()
        model.eval()

        train_loader, val_loader, test_loader = self.get_data()
        dataloader = train_loader + val_loader + test_loader

        with torch.no_grad():
            max_edges_year = []
            for g, _ in dataloader:
                ccodes = g.edata['edge_ccode'].cpu().numpy()
                years = np.full((ccodes.shape[0],), g.year.item(), dtype=int)
                _, att_w = model((g, _))
                weights = att_w[1].cpu().numpy()
                weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-8)
                
                name_src = [self.convert_ccode_to_name(c) for c in ccodes[:, 0]]
                name_dst = [self.convert_ccode_to_name(c) for c in ccodes[:, 1]]

                max_edges_year.append(pd.DataFrame({
                    'year': years,
                    'ccode_src': ccodes[:, 0],
                    'ccode_dst': ccodes[:, 1],
                    'name_src': name_src,
                    'name_dst': name_dst,
                    'weight': weights.squeeze()
                }))

        edges_df = pd.concat(max_edges_year).sort_values(by='year').reset_index(drop=True)
        idx_max = edges_df.groupby('year')['weight'].idxmax()
        top_edges_df = edges_df.loc[idx_max].reset_index(drop=True)

        top_edges_df['dyadid'] = (1000 * top_edges_df['ccode_src'] + top_edges_df['ccode_dst']).astype(int)
        top_edges_df.to_csv(self.folder + 'top_edge_max_weight.csv', index=False)

        top_k_edges = edges_df.groupby('year').apply(lambda x: x.nlargest(top_k, 'weight')).reset_index(drop=True)
        top_k_edges['dyadid'] = (1000 * top_k_edges['ccode_src'] + top_k_edges['ccode_dst']).astype(int)
        top_k_edges.to_csv(self.folder + f'top_{top_k}_edges_max_weight.csv', index=False)

        return top_k_edges

 

    def visualize_subgraphs_with_top_k_edges(self, years, top_k, use_safe_layout=True):
    
        print(f"[INFO] Starting visualization for top {top_k} edges and years {years}...")
        
        all_edges = pd.read_csv(self.graphs + 'edges.csv')
        file_csv = self.folder + f'top_{top_k}_edges_max_weight.csv'
        top_edges_df = pd.read_csv(file_csv)

        edf = all_edges[['graph_id', 'src_id', 'dst_id', 'weight']].rename({'graph_id':'year'}, axis=1)
        edf = edf[edf['year'].isin(years)].reset_index(drop=True)

        edf['name_src'] = edf['src_id'].apply(self.convert_ccode_to_name)
        edf['name_dst'] = edf['dst_id'].apply(self.convert_ccode_to_name)
        edf.drop(['src_id', 'dst_id'], axis=1, inplace=True)

        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.1)
        axs = [fig.add_subplot(gs[i]) for i in range(4)]

        for idx, year in enumerate(years):
            ax_attention = axs[idx]
            year_df = top_edges_df[top_edges_df['year'] == year]
            G = nx.DiGraph()

            for _, row in year_df.iterrows():
                G.add_edge(row['name_src'], row['name_dst'], weight=row['weight'])

            try:
                if use_safe_layout:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.spring_layout(G)
            except Exception as e:
                print(f"[WARN] Layout error for year {year}: {e}. Using circular layout.")
                pos = nx.circular_layout(G)

            nx.draw(G, pos, with_labels=True, node_size=1000,
                    node_color='lightblue', font_size=8,
                    font_weight='bold', arrows=True, width=1.,
                    ax=ax_attention, edge_color='black')

            try:
                edge_labels = {(row['name_src'], row['name_dst']): f"{row['weight']:.5f}" for _, row in year_df.iterrows()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',
                                            font_size=9, ax=ax_attention)
            except Exception as e:
                print(f"[WARN] Failed to draw edge labels for year {year}: {e}")

            ax_attention.set_title(f"Attention-Weighted SubGraphs - Year {year}", fontsize=12)
            ax_attention.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax_attention.transAxes,
                                                    color="black", fill=False, lw=2.))

        for idx, year in enumerate(years):
            ax_original = axs[idx + 2]

            year_df = top_edges_df[top_edges_df['year'] == year]
            edge_df = edf[(edf['year'] == year) &
                        (edf['name_src'].isin(year_df['name_src'])) &
                        (edf['name_dst'].isin(year_df['name_dst']))]

            edge_df = edge_df.merge(year_df[['name_src', 'name_dst']], on=['name_src', 'name_dst'], how='inner')
            if edge_df['weight'].max() > 0:
                edge_df['weight'] /= edge_df['weight'].max()

            G = nx.DiGraph()
            for _, row in edge_df.iterrows():
                G.add_edge(row['name_src'], row['name_dst'], weight=row['weight'])

            try:
                if use_safe_layout:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.kamada_kawai_layout(G)
            except Exception as e:
                print(f"[WARN] Layout error for year {year}: {e}. Using circular layout.")
                pos = nx.circular_layout(G)

            nx.draw(G, pos, with_labels=True, node_size=800,
                    node_color='lightblue', font_size=7,
                    font_weight='bold', arrows=True, width=1.,
                    ax=ax_original, edge_color='black')

            try:
                edge_labels = {(row['name_src'], row['name_dst']): f"{row['weight']:.5f}" for _, row in edge_df.iterrows()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',
                                            font_size=7, ax=ax_original)
            except Exception as e:
                print(f"[WARN] Failed to draw edge labels for year {year}: {e}")

            ax_original.set_title(f"IO-Weighted SubGraphs - Year {year}", fontsize=12)
            ax_original.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax_original.transAxes,
                                                    color="black", fill=False, lw=2.))

        plt.suptitle('IO-Weighted and Attention-Weighted SubGraphs', fontsize=20)
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"[WARN] tight_layout() failed: {e}")

        file_png = self.folder + f'plots/subgraphs_top_{top_k}_edges_{years[0]}_{years[1]}.png'
        try:
            plt.savefig(file_png, dpi=150, bbox_inches='tight')
            print(f"[INFO] Plot saved to: {file_png}")
        except Exception as e:
            print(f"[ERROR] Failed to save plot: {e}")

        plt.close()
        gc.collect()


    def uploading_persistence_entropy(self):
        first_year = self.first_year
        last_year = self.last_year
        path0 = self.data_homology + f'Entropy_hom_H_012_{first_year}_{last_year}.csv'
        entropy_values = pd.read_csv(path0, usecols = ['H_1'])
        return entropy_values

    def plot_weights_silhouette_mids_entropy(self):
        n_years = self.args['nyears']

        file_csv = self.folder + 'top_edge_max_weight.csv'
        top_df = pd.read_csv(file_csv)
        max_attw = top_df['weight'].values.tolist()

        fileyi = self.data_prepared +'yi.nc'
        yi = xr.open_dataset(fileyi).to_dataframe()
        midsDict = {year: yi.loc[yi['year'] == year, 'disp'].sum() for year in n_years}
        mids_values = [midsDict[year] for year in n_years]
        mids_max = np.max(np.array(mids_values))
        mids_values = [1 - mid/mids_max for mid in mids_values]

        entropy_values = self.uploading_persistence_entropy()
        max_ent = entropy_values.max()
        entropy_values = [1. - e/max_ent for e in list(entropy_values.values) ]

        max_attw = list(self.smooth_with_splines(max_attw))
        mids_values = list(self.smooth_with_splines(mids_values))
        entropy_values = self.smooth_with_splines(entropy_values)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(n_years, max_attw, color='blue', marker='o',
                 label='Max Attention Weights', linewidth = 2)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Max Attention Weights', color='blue', fontsize = 12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, max(max_attw) * 1.1)
        ax1.legend(loc = 'upper center' )

        ax2 = ax1.twinx()
        ax2.plot(n_years, entropy_values, color='green', marker='s',
                 label='Entropy', linewidth = 2)
        ax2.set_ylabel('1 - Entropy/MaxEnt', color='green',fontsize = 12)
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, np.max(entropy_values) * 1.1)
        ax2.legend(loc = 'upper right')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(n_years, mids_values, color='red', marker='^', linewidth = 2,
                 label='MIDs')
        ax3.set_ylabel('1 - MIDs/Max_MIDs', color='red', fontsize = 12)
        ax3.tick_params(axis='y', labelcolor='red')
        ax3.set_ylim(0, max(mids_values) * 1.1)
        ax3.legend(loc = 'upper left')

        plt.title('Max Attention Weights, Persistence Entropy, and MIDs', fontsize = 14)
        fig.tight_layout()
        file_png = self.folder + 'plots/max_weights_entropy_mids.png'
        plt.savefig(file_png)
        plt.close()
        gc.collect()

    
    def all_plots(self):
        
        self.plot_weights_probabilities()
        print('---------------------------------- Finished  plot_weights_probabilities')
        
        self.plot_silhouette_scores()   
        print('---------------------------------- Finished  plot_silhouette_scores')

        self.find_top_important_edges()
        print('---------------------------------- Finished  find_top_important_edges')

        years = [1985,2010]
        top_k = 8
        self.visualize_subgraphs_with_top_k_edges(years,top_k)
        print('---------------------------------- Finished  visualize_subgraphs_with_top_k_edges')

        self.plot_weights_silhouette_mids_entropy()
        print('---------------------------------- Finished  plot_weights_silhouette_mids_entropy')
        plt.close('all')