import torch
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


from .graphsdatamodule import GraphsDataModule

class GraphClusterAnalysis():
    def __init__(self, args):
        self.args = args
        self.phases = ['train','val','test']
        self.model_path = args['folder'] + 'GNN_models/model_path.ckpt'
        self.models_folder = args['models_folder']
        self.out_channels = args['out_channels']
        self.data_original = args['data_original']
        self.folder = args['folder'] + 'plots/'

   
    def get_data(self):    
        data_module = GraphsDataModule(self.args)
        data_module.setup()    
        return (
            data_module.train_data,
            data_module.val_data,
            data_module.test_data,
            )

    def simple_kmeans(self, X, n_clusters=4, n_iter=1000, seed=42):
        np.random.seed(seed)
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centers = X[indices].copy()

        for _ in range(n_iter):
            distances = self.compute_euclidean_distances(X, centers)
            labels = distances.argmin(axis=1)

            for k in range(n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    centers[k] = members.mean(axis=0)

        return centers, labels

    def compute_euclidean_distances(self, X, centers):
        X_norm = (X ** 2).sum(axis=1).reshape(-1, 1)
        C_norm = (centers ** 2).sum(axis=1).reshape(1, -1)
        cross_term = X @ centers.T
        dists_squared = X_norm + C_norm - 2 * cross_term
        dists_squared = np.maximum(dists_squared, 0.0)
        return np.sqrt(dists_squared)

    def analyze_probabilities(self):
        graph_embeddings = {'train': [], 'val': [], 'test': []}
        years = {'train': [], 'val': [], 'test': []}

        train_loader, val_loader, test_loader = self.get_data()
        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        for phase in self.phases:
            checkpoint_path = f"{self.models_folder}GNN_outputs_{phase}_last_epoch.ckpt"
            outs = torch.load(checkpoint_path)

            for out, data in zip(outs, dataloaders[phase]):
                g, _ = data
                year_tensor = torch.full((out.size(0),), g.year.item(), dtype=torch.long)
                graph_embeddings[phase].append(out)
                years[phase].append(year_tensor)

        output = torch.cat([torch.cat(graph_embeddings[phase], dim=0) for phase in self.phases], dim=0)
        years_combined = torch.cat([torch.cat(years[phase], dim=0) for phase in self.phases], dim=0)

        graph_embeddings_np = output.detach().cpu().numpy() #.astype(np.float64)
        graph_years = years_combined.detach().cpu().numpy()

        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(graph_embeddings_np)
        
        distances = pairwise_distances(graph_embeddings_np, kmeans.cluster_centers_)

        graph_probabilities = np.exp(-distances) 
        graph_probabilities = graph_probabilities / graph_probabilities.sum(axis=1, keepdims=True) 


        return graph_probabilities, graph_years


    def plot_probabilities(self):

        n_clusters = 4
        probabilities, years = self.analyze_probabilities()
        years = np.array(years).astype(int).flatten()

        unique_years = np.sort(np.unique(years))
        num_clusters = probabilities.shape[1]
        year_probabilities = np.zeros((len(unique_years), num_clusters))

        for i, year in enumerate(unique_years):
            indices = (years == year)
            year_probabilities[i] = np.nanmean(probabilities[indices], axis=0)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.stackplot(unique_years, year_probabilities.T, 
                    labels=[f'Cluster {i}' for i in range(num_clusters)], 
                    alpha=0.7)

        ax.set_xlabel('Year')
        ax.set_ylabel('Probability')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=n_clusters, 
                   bbox_to_anchor=(0.5, -0.025), fontsize=14)

        plt.suptitle('Cluster Membership Probabilities', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  

        plt.savefig(self.folder + 'train_test_class_probabilities_proportions_over_years.png', 
                    dpi = 300, bbox_inches='tight')
        plt.close('all')
        gc.collect()

    