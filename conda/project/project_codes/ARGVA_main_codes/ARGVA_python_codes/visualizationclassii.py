import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.stats import kde
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin
from torch_geometric.utils import subgraph
import xarray as xr
import torch

from .argvamodel import ARGVAModel
from .createdatamodule import CreateDataModule


class VisualizationClassII():
    """ 
    This class creates plots for all years from first_year to last_year as well as
    as for specific years 
    """
    def __init__(self, args):
        self.args = args
        self.n_epochs = args['n_epochs'] 
        self.folder = args['folder']
        self.data_original = args['data_original']
        self.data_prepared = args['data_prepared']

    def loading_model(self, year):
        checkpoint_path = self.args['models_folder']+ f'/model_{year}_path.ckpt'
        model = ARGVAModel.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                                      year=year,args=self.args)
        return model
        
    """ Convert ccode to state name """
    def convert_ccode_to_name(self, ccode):
        args = self.args
        countries = pd.read_csv(self.data_original + 'COW_country_codes.csv')[['ccode', 'stateabb']]
        name = countries.loc[countries['ccode'] == ccode, :]['stateabb'].values[0]
        return name

    def get_data(self, year):                                      
        datamodule = CreateDataModule(self.args, year)
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

    def z_parameterize(self, model, x, edge_index, edge_attr):
        mu, logvar,_ = model(x, edge_index, edge_attr)
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        z = mu + std * eps  
        return z

    def compute_node_embedding(self, year, data, mask, key):
        model = self.loading_model(year)
        edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                                                               
        embeddings = self.z_parameterize(model.encoder, 
                                         data.x[mask],
                                         edge_index_masked, 
                                         edge_attr_masked)
        
        return embeddings

    def count_points_per_cluster(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        points_per_cluster = dict(zip(unique, counts))
        return points_per_cluster


    def average_proportion_nodes_in_kmeans_clusters(self):
        """ Average proportion nodes in kmeans clusters """
        
        args = self.args
        nyears = args['nyears']
       
        fig, ax = plt.subplots(figsize=(12, 8))
        silhouette_scores = {'train': [], 'test': []}
        silhouette_cluster_scores = {'train': {f'Cluster_{j}': [] for j in range(4)},
                                     'test': {f'Cluster_{j}': [] for j in range(4)}}
        
        combined_ar_kcc_list = []
        combined_silhouette_values = []
    
        for i, key in enumerate(['train', 'test']):
            
            for year in tqdm(nyears): 
                model = self.loading_model(year)
                model.eval()
                dataloader = self.get_data(year)[key]
                
                emb_list, ya_list = [], []
                for data in dataloader:
                    mask = getattr(data, f"{key}_mask").bool()
                    edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                    
                    emb = self.compute_node_embedding(year, data, mask, key)
                    emb_list.append(emb.detach().numpy())
                    ya_list.append(data.y[mask].detach().numpy())
    
                emb_array = np.vstack(emb_list)
                y = np.hstack(ya_list)
                pca = PCA(n_components=2) 
                reduced_data = pca.fit_transform(emb_array)
                     
                n_clusters = 4
                k_means = KMeans(init="k-means++", 
                                 n_clusters=n_clusters, 
                                 max_iter=2000, 
                                 random_state=0, 
                                 n_init='auto')
                k_means.fit(reduced_data)
                
                k_means_cluster_centers = k_means.cluster_centers_
                k_means_labels = pairwise_distances_argmin(reduced_data, k_means_cluster_centers)
                
                kmeans_cluster_counts = self.count_points_per_cluster(k_means_labels)
                combined_ar_kcc_list.append(np.array(list(kmeans_cluster_counts.values())))
    
                silhouette_values = silhouette_samples(reduced_data, k_means_labels)
                combined_silhouette_values.extend(silhouette_values)
    
                for j in range(n_clusters):
                    cluster_silhouette_values = silhouette_values[k_means_labels == j]
                    silhouette_cluster_scores[key][f'Cluster_{j}'].append(np.mean(cluster_silhouette_values))
    
                silhouette_avg = silhouette_score(reduced_data, k_means_labels)
                silhouette_scores[key].append(silhouette_avg)
    
        labels = [f'Cluster_{i}' for i in range(4)]
        
        combined_ar_kcc = np.vstack(combined_ar_kcc_list)
        df_kcc = pd.DataFrame(data=combined_ar_kcc, columns=labels)
        df_kcc = pd.concat([pd.DataFrame({'year':list(nyears) * 2}), df_kcc], axis=1)
        
        df_kcc_proportions = df_kcc.copy()
        total_nodes_per_year = df_kcc[labels].sum(axis=1)
        for label in labels:
            df_kcc_proportions[label] = df_kcc[label] / total_nodes_per_year
        
        df_long = pd.melt(df_kcc_proportions, id_vars=['year'], value_vars=labels, 
                          var_name='label', value_name='proportion')
        
        df_mean_proportions = df_long.groupby(['year', 'label'], 
                                              as_index=False)['proportion'].mean()
        
        df_pivot = df_mean_proportions.pivot(index='year', 
                                             columns='label', values='proportion')
            
        ax.stackplot(df_pivot.index, df_pivot.T, labels=labels, alpha=0.6)
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Mean Proportion of Nodes', fontsize=14)
        ax.legend()
        
        fileyi = self.data_prepared + 'yi.nc'
        yi = xr.open_dataset(fileyi).to_dataframe()
        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()} 
                    for year in nyears}
        mids = [midsDict[year]['mids'] for year in nyears]
        max_mids = np.max(np.array(mids))
        mids = [mid/max_mids for mid in mids]
        
        mids = self.smooth_with_splines(mids)

        ax2 = ax.twinx()
        color = 'k'
        ax2.set_ylabel('Normalized MIDs', color=color, fontsize=18)
        ax2.plot(nyears, mids, color=color, label='MIDs')
        ax2.tick_params(axis='y', labelcolor=color, size=10)
        ax2.spines['right'].set_position(('outward', 60))  

        plt.suptitle('Mean Proportions of Nodes per Cluster', fontsize=18)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        path1 = 'classifications/average_proportions_of_nodes_per_cluster_per_year_combined.png'
        plt.savefig(args['folder'] + path1)
        plt.close()
    
        return

    def smooth_with_splines(self, y):
        y = np.array(y)
        x = np.arange(y.shape[0])
        spline = UnivariateSpline(x, y, s=10,k=5)
        y_smooth = spline(x)
        return y_smooth
    
    def silhouette_scores(self):
        args = self.args
        nyears = args['nyears']
    
        silhouette_scores = {'train': [], 'test': []}
        silhouette_cluster_scores = {'train': {f'Cluster_{j}': [] for j in range(4)},
                                    'test': {f'Cluster_{j}': [] for j in range(4)}}
        
        for i, key in enumerate(['train', 'test']):
            
            for year in tqdm(nyears): 
                model = self.loading_model(year)
                model.eval()
                dataloader = self.get_data(year)[key]
                
                emb_list, ya_list = [], []
                for data in dataloader:
                    mask = getattr(data, f"{key}_mask").bool()
                    edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                    
                    emb = self.compute_node_embedding(year, data, mask, key)
                    emb_list.append(emb.detach().numpy())
                    ya_list.append(data.y[mask].detach().numpy())

                emb_array = np.vstack(emb_list)
                y = np.hstack(ya_list)
                pca = PCA(n_components=2) 
                reduced_data = pca.fit_transform(emb_array)
                    
                n_clusters = 4
                k_means = KMeans(init="k-means++", 
                                n_clusters=n_clusters, 
                                max_iter=2000, 
                                random_state=0, 
                                n_init='auto')
                k_means.fit(reduced_data)
                
                k_means_cluster_centers = k_means.cluster_centers_
                k_means_labels = pairwise_distances_argmin(reduced_data, k_means_cluster_centers)
                
                kmeans_cluster_counts = self.count_points_per_cluster(k_means_labels)

                silhouette_values = silhouette_samples(reduced_data, k_means_labels)
                for j in range(n_clusters):
                    cluster_silhouette_values = silhouette_values[k_means_labels == j]
                    silhouette_cluster_scores[key][f'Cluster_{j}'].append(np.mean(cluster_silhouette_values))

                silhouette_avg = silhouette_score(reduced_data, k_means_labels)
                silhouette_scores[key].append(silhouette_avg)

        labels = [f'Cluster_{i}' for i in range(4)]

        fig, ax = plt.subplots(figsize=(12, 7))

        for key in ['train', 'test']:
            df_silhouette_scores = pd.DataFrame(silhouette_cluster_scores[key])
            df_silhouette_scores['year'] = nyears
            df_silhouette_scores = df_silhouette_scores.set_index('year')
            
            df_silhouette_scores_normalized = df_silhouette_scores.div(df_silhouette_scores.sum(axis=1), 
                                                                    axis=0)
            
            df_long = pd.melt(df_silhouette_scores_normalized.reset_index(), id_vars=['year'], 
                            value_vars=[f'Cluster_{i}' for i in range(n_clusters)], 
                            var_name='label', value_name='proportion')
            
            df_mean_proportions = df_long.groupby(['year', 'label'], 
                                              as_index=False)['proportion'].mean()
        
            df_pivot = df_mean_proportions.pivot(index='year', 
                                                columns='label', values='proportion')
            
            ax.stackplot(df_pivot.index, df_pivot.T, 
                        labels=[f'{key.capitalize()} Cluster_{i}' for i in range(n_clusters)],
                        alpha=0.6)
            
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Silhouette Score Proportion', fontsize=14)
        ax.set_title('Normalized Silhouette Scores Proportions (Train and Test)', fontsize=18)
        
        plt.legend(loc='upper right')
        plt.tight_layout()

        path3 = 'classifications/normalized_silhouette_score_proportions_per_cluster_per_year.png'
        plt.savefig(args['folder'] + path3)
        plt.close()

        return

    def node_density_scores(self):
        
        args = self.args
        nyears = args['nyears']
    
        node_density_cluster_scores = {'train': {f'Cluster_{j}': [] for j in range(4)},
                                    'test': {f'Cluster_{j}': [] for j in range(4)}}
        
        for key in ['train', 'test']:
            for year in tqdm(nyears): 
                model = self.loading_model(year)
                model.eval()
                dataloader = self.get_data(year)[key]
                
                emb_list, ya_list = [], []
                for data in dataloader:
                    mask = getattr(data, f"{key}_mask").bool()
                    edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                    
                    emb = self.compute_node_embedding(year, data, mask, key)
                    emb_list.append(emb.detach().numpy())
                    ya_list.append(data.y[mask].detach().numpy())

                emb_array = np.vstack(emb_list)
                y = np.hstack(ya_list)
                pca = PCA(n_components=2) 
                reduced_data = pca.fit_transform(emb_array)
                    
                n_clusters = 4
                k_means = KMeans(init="k-means++", 
                                n_clusters=n_clusters, 
                                max_iter=2000, 
                                random_state=0, 
                                n_init='auto')
                k_means.fit(reduced_data)
                
                k_means_cluster_centers = k_means.cluster_centers_
                k_means_labels = pairwise_distances_argmin(reduced_data, k_means_cluster_centers)
                
                cluster_counts = np.bincount(k_means_labels, minlength=n_clusters)
                total_nodes = np.sum(cluster_counts)
                
                cluster_densities = cluster_counts / total_nodes
                
                for j in range(n_clusters):
                    node_density_cluster_scores[key][f'Cluster_{j}'].append(cluster_densities[j])

        combined_node_density_scores = []
        for key in ['train', 'test']:
            for j in range(4):
                df_temp = pd.DataFrame({
                    'Year': nyears,
                    'Node Density': node_density_cluster_scores[key][f'Cluster_{j}'],
                    'Cluster': f'Cluster_{j}',
                    'Type': key.capitalize()
                })
                combined_node_density_scores.append(df_temp)

        df_combined = pd.concat(combined_node_density_scores)

        fig, axs = plt.subplots(1,2,figsize=(14, 7))
        axs = axs.flatten()
        for ax, data_type in zip(axs,['train', 'test']):
            df_plot = df_combined[df_combined['Type'] == data_type.capitalize()]
            df_pivot = df_plot.pivot(index='Year', columns='Cluster', values='Node Density')
            ax.stackplot(df_pivot.index, df_pivot.T, labels=[f'Cluster_{i}' for i in range(n_clusters)], 
                         alpha=0.6)
            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('Node Density', fontsize=14)
            ax.set_title(f'Node Density in Clusters ({data_type.capitalize()})', fontsize=16)
            ax.legend(loc='upper right')

        plt.tight_layout()

        path3 = f'classifications/node_density_stackplot.png'
        plt.savefig(args['folder'] + path3)
        plt.close()

        return
    
    def convert_ccode_to_name(self, ccode):
        countries = pd.read_csv(self.data_original + 'COW_country_codes.csv')[['ccode', 'stateabb']]
        name = countries.loc[countries['ccode'] == ccode, :]['stateabb'].values[0]
        return name
    
    def record_nodes_in_clusters(self):
        args = self.args
        nyears = args['nyears']

        all_rows = []
        for key in ['train', 'test']:
            for year in tqdm(nyears): 
                model = self.loading_model(year)
                model.eval()
                
                data_module = CreateDataModule(args,year)
                data_module.setup()

                dataloader = self.get_data(year)[key]
                
                emb_list, ya_list = [], []
                for data in dataloader:
                    mask = getattr(data, f"{key}_mask").bool()
                    
                    emb = self.compute_node_embedding(year, data, mask, key)
                    emb_list.append(emb.detach().numpy())
                    ya_list.append(data.y[mask].detach().numpy())

                emb_array = np.vstack(emb_list)
                y = np.hstack(ya_list)
                pca = PCA(n_components=2) 
                reduced_data = pca.fit_transform(emb_array)
                    
                n_clusters = 4
                k_means = KMeans(init="k-means++", 
                                n_clusters=n_clusters, 
                                max_iter=2000, 
                                random_state=0, 
                                n_init='auto')
                k_means.fit(reduced_data)
                
                k_means_labels = k_means.labels_

                node_indices = data_module.get_node_indices(split=key)[year]
                node_ccodes = data_module.get_node_ccodes(split=key)[year]

                for j in range(n_clusters):
                    cluster_nodes = np.where(k_means_labels == j)[0]
                    nodes = node_indices[cluster_nodes]
                    ccodes = node_ccodes[cluster_nodes].numpy()

                    for node, ccode in zip(nodes, ccodes):
                        all_rows.append({
                            'Year': year,
                            'Node': node,
                            'Ccode': ccode,
                            'Statename': self.convert_ccode_to_name(ccode),
                            'Cluster': f'Cluster_{j}',
                            'Type': key.capitalize()
                        })

        df = pd.DataFrame(all_rows)

        path = f'classifications/nodes_and_ccodes_in_clusters.csv'
        df.to_csv(args['folder'] + path, index=False)

        plt.figure(figsize=(14, 8))
        Statename = 'USA'

        da = df.loc[df['Statename'] == Statename, :]
        data = da.groupby(['Year', 'Cluster']).size().reset_index(name='Count')\
        
        data_pivot = data.pivot(index='Year', columns='Cluster', values='Count').fillna(0)
        # Calculate the probabilities by dividing each cluster count by the row sum
        data_prob = data_pivot.div(data_pivot.sum(axis=1), axis=0)

        plt.figure(figsize=(14, 8))
        plt.stackplot(data_prob.index, data_prob.T, labels=data_prob.columns, alpha=0.7)

        plt.title(f'Probability Distribution of {Statename} in Clusters', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.legend(title='Cluster', loc='upper left')
        plt.tight_layout()
        plt.close()

        return df

    def assign_clusters_to_nodes(self, dataloader, key, year, n_clusters=4):
        model = self.loading_model(year)
        model.eval()
        
        node_embeddings = []
        node_probabilities = []
        
        with torch.no_grad():
            for data in dataloader:
                mask = getattr(data, f"{key}_mask").bool()
                emb = self.compute_node_embedding(year, data, mask, key)
                node_embeddings.append(emb.detach().numpy())

        node_embeddings = np.concatenate(node_embeddings)
    
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(node_embeddings)
        
        distances = pairwise_distances(node_embeddings, kmeans.cluster_centers_)

        node_probabilities = np.exp(-distances) 
        node_probabilities = node_probabilities / node_probabilities.sum(axis=1, keepdims=True) 

        return node_probabilities

    def extract_ccodes(self, dataloader):
        ccodes = []
        for data in dataloader:
            ccodes.append(data.node_ccodes.cpu().numpy())
        return np.concatenate(ccodes)

    def extract_years(self, dataloader):
        years = []
        for data in dataloader:
            years.append(data.year.detach().numpy())  
        return np.hstack(years)

    def analyze_node_probabilities(self, year):
        dataloaders= self.get_data(year)
        keys = ['train','val','test']
        dataloaders = [dataloaders[key] for key in keys]

        all_probabilities = []
        all_ccodes = []
        all_names = []
        all_years = []

        for dataloader,key in zip(dataloaders,keys):
            node_probabilities = self.assign_clusters_to_nodes(dataloader,key, year)
            node_ccodes = self.extract_ccodes(dataloader)
            node_years = self.extract_years(dataloader)
            node_names = [self.convert_ccode_to_name(ccode) for ccode in node_ccodes]

            all_probabilities.append(node_probabilities)
            all_ccodes.append(node_ccodes)
            all_names.append(node_names)
            all_years.append(node_years)

        all_probabilities = np.concatenate(all_probabilities)
        all_ccodes = np.concatenate(all_ccodes)
        all_names = np.concatenate(all_names)
        all_years = np.concatenate(all_years)

        return all_probabilities, all_ccodes, all_names, all_years

    def plot_node_probabilities(self):
        args = self.args
        n_clusters = 4 
        all_probabilities, all_ccodes, all_names, all_years = [], [], [], []
    
        for year in args['nyears']:
            probabilities, ccodes, names, years = self.analyze_node_probabilities(year)
            all_probabilities.append(probabilities)
            all_ccodes.append(ccodes)
            all_names.append(names)
            all_years.append(years)
    
        all_probabilities = np.concatenate(all_probabilities)
        all_ccodes = np.concatenate(all_ccodes)
        all_names = np.concatenate(all_names)
        all_years = np.concatenate(all_years)
    
        all_rows = []
        for i in range(len(all_probabilities)):
            all_rows.append({
                'Year': all_years[i],
                'Ccode': all_ccodes[i],
                'Statename': all_names[i],
                'Cluster_Probabilities': all_probabilities[i] 
            })
    
        df = pd.DataFrame(all_rows)
       
        states = ['USA', 'CHN', 'UKG','NIG','BRA','RUS'] 
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
    
        labels = [f'Cluster {i}' for i in range(n_clusters)]
    
        for ax, state in zip(axs, states):
            df_filtered = df[df['Statename'] == state]
            unique_years = np.sort(np.unique(df_filtered['Year']))
            year_probabilities = np.zeros((len(unique_years), n_clusters))
    
            for i, year in enumerate(unique_years):
                year_data = df_filtered[df_filtered['Year'] == year]
                year_probs = np.vstack(year_data['Cluster_Probabilities'].values)
        
                if year_probs.shape[1] < n_clusters:
                    padded_probs = np.zeros((year_probs.shape[0], n_clusters))
                    padded_probs[:, :year_probs.shape[1]] = year_probs
                    year_probs = padded_probs
                elif year_probs.shape[1] > n_clusters:
                    year_probs = year_probs[:, :n_clusters]
    
                year_probabilities[i] = np.mean(year_probs, axis=0)
        
            ax.stackplot(unique_years, year_probabilities.T, labels=labels, alpha=0.7)
            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('Probability', fontsize=14)
            ax.set_title(f'{state}', fontsize=15)
    
        # Creating a single common legend below the plot box
        fig.legend(labels=labels, loc='lower center', ncol=n_clusters, fontsize=12, bbox_to_anchor=(0.5, -0.05))
    
        plt.suptitle('Cluster Membership Probabilities', fontsize=18)  
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
        plt.savefig(args['folder'] + 'multiple_states_cluster_probabilities_over_years.png', bbox_inches='tight')
        plt.close()
                    
        return 

    
    def multiple_cluster_analyses(self):

        args = self.args
        nyears = args['nyears']
        n_clusters = 4  
        
        cluster_centroids = {i: [] for i in range(n_clusters)}
        intra_cluster_distances = {i: [] for i in range(n_clusters)}
        inter_cluster_distances = []
        cluster_compactness = {i: [] for i in range(n_clusters)}
        cluster_purity = {i: [] for i in range(n_clusters)}
        
        for year in tqdm(nyears): 
            model = self.loading_model(year)
            model.eval()
            dataloader = self.get_data(year)['test']
            
            emb_list, ya_list = [], []
            for data in dataloader:
                mask = getattr(data, "test_mask").bool()
                edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                
                emb = self.compute_node_embedding(year, data, mask, 'test')
                emb_list.append(emb.detach().numpy())
                ya_list.append(data.y[mask].detach().numpy())

            emb_array = np.vstack(emb_list)
            y = np.hstack(ya_list)
            pca = PCA(n_components=2) 
            reduced_data = pca.fit_transform(emb_array)
                
            k_means = KMeans(init="k-means++", 
                            n_clusters=n_clusters, 
                            max_iter=2000, 
                            random_state=0, 
                            n_init='auto')
            k_means.fit(reduced_data)
            
            k_means_labels = k_means.labels_
            k_means_cluster_centers = k_means.cluster_centers_
            
            # 1. Cluster Centroid Movement (Store centroids for stack plot)
            for i in range(n_clusters):
                cluster_centroids[i].append(k_means_cluster_centers[i])
            
            # 2. Cluster Cohesion (Intra-Cluster Distance)
            for i in range(n_clusters):
                distances = np.mean(pairwise_distances(reduced_data[k_means_labels == i]))
                intra_cluster_distances[i].append(distances)
        
            # 3. Cluster Separation (Inter-Cluster Distance)
            inter_cluster_distances_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    distance = np.linalg.norm(k_means_cluster_centers[i] - k_means_cluster_centers[j])
                    inter_cluster_distances_matrix[i, j] = distance
                    inter_cluster_distances_matrix[j, i] = distance
            inter_cluster_distances.append(inter_cluster_distances_matrix)
            
            # 4. Cluster Compactness
            for i in range(n_clusters):
                compactness = np.mean(np.linalg.norm(reduced_data[k_means_labels == i] -k_means_cluster_centers[i], axis=1))
                cluster_compactness[i].append(compactness)
            
            # 5. Cluster Purity (if ground truth labels are available)
            for i in range(n_clusters):
                majority_label_count = np.bincount(y[k_means_labels == i]).max()
                purity = majority_label_count / np.sum(k_means_labels == i)
                cluster_purity[i].append(purity)
        
        # Normalization: Convert all metrics to proportions
        def normalize(data):
            data = np.array(data)
            return data / data.sum(axis=0, keepdims=True)
    
        cohesion_stack = normalize([intra_cluster_distances[i] for i in range(n_clusters)])
        separation_stack = normalize([[dist[i].mean() for dist in inter_cluster_distances] 
                                      for i in range(n_clusters)])
        compactness_stack = normalize([cluster_compactness[i] for i in range(n_clusters)])
        purity_stack = normalize([cluster_purity[i] for i in range(n_clusters)])
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()

        # 2. Cluster Cohesion Over Time
        axs[0].stackplot(nyears, cohesion_stack, labels=[f'Cluster {i}' for i in range(n_clusters)], 
                         alpha=0.6)
        axs[0].set_xlabel('Year')
        axs[0].set_ylabel('Proportion (Intra-Cluster Distance)')
        axs[0].set_title('Cluster Cohesion Over Time')
        axs[0].legend()

        # 3. Cluster Separation Over Time
        axs[1].stackplot(nyears, separation_stack, labels=[f'Cluster {i}' for i in range(n_clusters)], 
                         alpha=0.6)
        axs[1].set_xlabel('Year')
        axs[1].set_ylabel('Proportion (Inter-Cluster Distance)')
        axs[1].set_title('Cluster Separation Over Time')
        axs[1].legend()

        # 4. Cluster Compactness Over Time
        axs[2].stackplot(nyears, compactness_stack, labels=[f'Cluster {i}' for i in range(n_clusters)], 
                         alpha=0.6)
        axs[2].set_xlabel('Year')
        axs[2].set_ylabel('Proportion (Compactness)')
        axs[2].set_title('Cluster Compactness Over Time')
        axs[2].legend()
    
        # 5. Cluster Purity Over Time
        axs[3].stackplot(nyears, purity_stack, labels=[f'Cluster {i}' for i in range(n_clusters)], 
                         alpha=0.6)
        axs[3].set_xlabel('Year')
        axs[3].set_ylabel('Proportion (Purity)')
        axs[3].set_title('Cluster Purity Over Time')
        axs[3].legend()

        plt.tight_layout()
        
        path = 'classifications/multiple_cluster_analyses_test_data.png'
        plt.savefig(args['folder'] + path)
        plt.close()

        return
    
    
    def cluster_cohesion_proportions_over_time(self):
        args = self.args
        nyears = args['nyears']
        
        n_clusters = 4  
        cohesion_scores_over_time = {i: [] for i in range(n_clusters)}

        for year in tqdm(nyears): 
            model = self.loading_model(year)
            model.eval()
            dataloader = self.get_data(year)['test']
            
            emb_list, ya_list = [], []
            for data in dataloader:
                mask = getattr(data, "test_mask").bool()
                edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                
                emb = self.compute_node_embedding(year, data, mask, 'test')
                emb_list.append(emb.detach().numpy())
                ya_list.append(data.y[mask].detach().numpy())

            emb_array = np.vstack(emb_list)
            y = np.hstack(ya_list)
            pca = PCA(n_components=2) 
            reduced_data = pca.fit_transform(emb_array)
                
            k_means = KMeans(init="k-means++", 
                            n_clusters=n_clusters, 
                            max_iter=2000, 
                            random_state=0, 
                            n_init='auto')
            k_means.fit(reduced_data)
            
            k_means_labels = k_means.labels_

            for i in range(n_clusters):
                cluster_points = reduced_data[k_means_labels == i]
                if len(cluster_points) > 1:
                    distances = np.mean(pairwise_distances(cluster_points))
                    cohesion_scores_over_time[i].append(distances)
                else:
                    cohesion_scores_over_time[i].append(0)  

        def normalize(data):
            data = np.array(data)
            return data / data.sum(axis=0, keepdims=True)

        cohesion_stack = normalize([cohesion_scores_over_time[i] for i in range(n_clusters)])

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.stackplot(nyears, cohesion_stack, labels=[f'Cluster {i}' for i in range(n_clusters)], 
                     alpha=0.6)
        ax.set_xlabel('Year')
        ax.set_ylabel('Proportion (Intra-Cluster Distance)')
        ax.set_title('Cluster Cohesion Over Time')
        ax.legend()

        path = 'classifications/cohesion_cluster_proportions_analysis_test_data.png'
        plt.savefig(args['folder'] + path)
        plt.close()

        return 
    
    def individual_cluster_cohesion_over_time(self):
        args = self.args
        nyears = args['nyears']
        
        n_clusters = 4  
        cohesion_scores_over_time = {i: [] for i in range(n_clusters)}

        for year in tqdm(nyears): 
            model = self.loading_model(year)
            model.eval()
            dataloader = self.get_data(year)['test']
            
            emb_list, ya_list = [], []
            for data in dataloader:
                mask = getattr(data, "test_mask").bool()
                edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                
                emb = self.compute_node_embedding(year, data, mask, 'test')
                emb_list.append(emb.detach().numpy())
                ya_list.append(data.y[mask].detach().numpy())

            emb_array = np.vstack(emb_list)
            y = np.hstack(ya_list)
            pca = PCA(n_components=2) 
            reduced_data = pca.fit_transform(emb_array)
                
            k_means = KMeans(init="k-means++", 
                            n_clusters=n_clusters, 
                            max_iter=2000, 
                            random_state=0, 
                            n_init='auto')
            k_means.fit(reduced_data)
            
            k_means_labels = k_means.labels_

            for i in range(n_clusters):
                cluster_points = reduced_data[k_means_labels == i]
                if len(cluster_points) > 1:
                    distances = np.mean(pairwise_distances(cluster_points))
                    cohesion_scores_over_time[i].append(distances)
                else:
                    cohesion_scores_over_time[i].append(0)  

        fileyi = self.data_prepared + 'yi.nc'
        yi = xr.open_dataset(fileyi).to_dataframe()
        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()} 
                    for year in nyears}
        mids = [midsDict[year]['mids'] for year in nyears]
        max_mids = np.max(np.array(mids))
        mids = [mid/max_mids for mid in mids]
        
        mids = self.smooth_with_splines(mids)

        fig, axs = plt.subplots(2, 2, figsize=(14, 14))  
        axs = axs.flatten()  
        
        for i,ax in zip(range(n_clusters),axs):
            cohesion = self.smooth_with_splines(cohesion_scores_over_time[i])
            line1, = ax.plot(nyears, cohesion, marker='o', label = 'Cohesion',
                    linestyle = '-.', linewidth = 2)
            ax.set_title(f'Cluster {i}', fontsize = 16)
            ax.set_xlabel('Year', fontsize = 14)
            ax.set_ylabel('Smoothed Cohesion', fontsize = 14)
            ax.grid(True)
           
            ax2 = ax.twinx()
            color = 'red'
            ax2.set_ylabel('Smoothed Normalized MIDs', color=color,fontsize = 14)
            line2, = ax2.plot(nyears,mids, color=color, label='MIDs', marker = 'd',
                     linestyle = '-', linewidth = 2)
            ax2.tick_params(axis='y', labelcolor=color, size=10)
            
            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels)

        plt.tight_layout()
        path = 'classifications/smoothed_individual_cohesion_cluster_analysis_test_data.png'
        plt.savefig(args['folder'] + path)
        plt.close()

        return

    def grouped_individual_cluster_cohesion_over_time(self):
        args = self.args
        nyears = args['nyears']
        
        n_clusters = 4  
        cohesion_scores_over_time = {i: [] for i in range(n_clusters)}

        for year in tqdm(nyears): 
            model = self.loading_model(year)
            model.eval()
            dataloader = self.get_data(year)['test']
            
            emb_list, ya_list = [], []
            for data in dataloader:
                mask = getattr(data, "test_mask").bool()
                edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                
                emb = self.compute_node_embedding(year, data, mask, 'test')
                emb_list.append(emb.detach().numpy())
                ya_list.append(data.y[mask].detach().numpy())

            emb_array = np.vstack(emb_list)
            y = np.hstack(ya_list)
            pca = PCA(n_components=2) 
            reduced_data = pca.fit_transform(emb_array)
                
            k_means = KMeans(init="k-means++", 
                            n_clusters=n_clusters, 
                            max_iter=2000, 
                            random_state=0, 
                            n_init='auto')
            k_means.fit(reduced_data)
            
            k_means_labels = k_means.labels_

            for i in range(n_clusters):
                cluster_points = reduced_data[k_means_labels == i]
                if len(cluster_points) > 1:
                    distances = np.mean(pairwise_distances(cluster_points))
                    cohesion_scores_over_time[i].append(distances)
                else:
                    cohesion_scores_over_time[i].append(0)  

        fileyi = self.data_prepared + 'yi.nc'
        yi = xr.open_dataset(fileyi).to_dataframe()
        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()} 
                    for year in nyears}
        mids = [midsDict[year]['mids'] for year in nyears]
        max_mids = np.max(np.array(mids))
        mids = [mid/max_mids for mid in mids]
        
        mids = self.smooth_with_splines(mids)

        fig, ax = plt.subplots(figsize=(12, 8))  
        colors = ['blue','red','green','magenta']
        markers = ['o','D','*','v']
        linestyles = ['--',':','-.', '-']
        for i in range(n_clusters):
            cohesion = self.smooth_with_splines(cohesion_scores_over_time[i])
            ax.plot(nyears, cohesion, marker=markers[i], label = f'Cluster {i}',
                     color = colors[i], linewidth = 2,linestyle = linestyles[i])
            ax.set_xlabel('Year', fontsize = 14)
            ax.set_ylabel('Smoothed Cohesion', fontsize = 14)
            ax.grid(True)
            ax.legend()
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        ax2 = ax.twinx()
        color = 'k'
        ax2.set_ylabel('Smoothed Normalized MIDs', color=color,fontsize = 14)
        line2, = ax2.plot(nyears,mids, color=color, label='MIDs', marker = '+',
                    linestyle = '-', linewidth = 2)
        ax2.tick_params(axis='y', labelcolor=color, size=10)
        ax2.legend()

        plt.title('Cluster Cohesion and Total MIDs over the Years', fontsize =18) 

        plt.tight_layout()
        path = 'classifications/grouped_smoothed_individual_cohesion_cluster_analysis_test_data.png'
        plt.savefig(args['folder'] + path)
        plt.close()

        return