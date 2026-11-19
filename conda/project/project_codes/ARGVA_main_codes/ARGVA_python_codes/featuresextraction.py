import pytorch_lightning as pl
import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, estimate_bandwidth,MeanShift
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.spatial import distance


from .createdatamodule import CreateDataModule
from .argvamodel import ARGVAModel

class FeaturesExtraction():

    def __init__(self, args):
        super(FeaturesExtraction, self).__init__()
        self.args = args
        self.first_year = args['first_year']
        self.last_year = args['last_year']
        self.folder = args['folder']
        self.models_folder = args['models_folder']
        self.n_epochs = args['n_epochs']
        self.data_homology = args['data_homology']

    def get_data(self,year):
        datamodule = CreateDataModule(self.args,year)
        datamodule.setup()

        dataloaders = {
            "train":datamodule.train_dataloader(),
            "val":datamodule.val_dataloader(),
            "test": datamodule.test_dataloader(),
            }
        return dataloaders

    def loading_model(self,year):

        checkpoint_path = self.models_folder+ f'model_{year}_path.ckpt'
        model = ARGVAModel.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                      year=year,args=self.args)
        return model


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

    def extracting_z(self, year, data,mask):

        model = self.loading_model(year)
        model.eval()

        edge_index_masked, edge_attr_masked = self.masking_data(data, mask)

        z = self.z_parameterize(model.encoder,
                                    data.x[mask],
                                    edge_index_masked,
                                    edge_attr_masked)
        return z

    def building_nodes_latent_features(self):
        zDict = {}
        keys =  ['train', 'val', 'test']
        for year in tqdm(self.args['nyears']):
            datamodule = CreateDataModule(self.args, year)
            datamodule.setup()

            keyDict = {}
            for key in keys:
                dataloader = self.get_data(year)[key]

                za_list,nodes_list = [],[]
                for data in dataloader:
                    mask = getattr(data, f"{key}_mask").bool()
                    z = self.extracting_z(year, data, mask)
                    z_batch = z.detach().numpy()
                    za_list.append(z_batch)
                    nodes_list.append(data.node_indices)
                nodes_indices = (torch.cat(nodes_list)).detach().numpy()
                z_array = np.vstack(za_list)

                z_pca = PCA(n_components=2).fit_transform(z_array)
                z_df = pd.DataFrame(z_pca,
                                    columns=[f'LatentFeature_{i}' for i in range(2)])
                z_df['nodes_indices'] = pd.DataFrame({'nodes_indices': nodes_indices})
                z_df['year'] = pd.DataFrame({'year': np.tile(year,z_df.shape[0])})
                z_df = z_df.iloc[:,[-1,-2,0,1]]
                keyDict[key] = z_df

                pathfilez = self.folder + f'latentspace/latentspace_csv/'
                pathfilez += f'ARGVA_{key}_latent_features_{year}.csv'
                z_df.to_csv(pathfilez, index=False)

            zDict[year] = keyDict

        return zDict

    def plot_latent_space(self):

        ys = self.args['specific_years']
        colors = ['red', 'green']
        keys = ['train', 'val','test']

        yDict = {}
        for year in ys:
            dataloaders = self.get_data(year)
            for key in keys:
                yd =  {}
                for key in keys:
                    dataloader = dataloaders[key]
                    y_list = []
                    for data in dataloader:
                        mask = getattr(data, f"{key}_mask").bool()
                        y_list.append(data.y[mask])

                    yd[key] = torch.concat(y_list).detach().numpy().astype(int).tolist()

            yDict[year] = yd

        """
        Plots PCA KDE for latentspace
        """
        zDict = self.building_nodes_latent_features()
        keys = ['train','test']
        disp = [1,0]
        for year in ys:
            z_dff , count_of_ones = {}, {}
            for key in keys:
                y_data = yDict[year][key]
                count_of_ones[key] = np.sum(y_data)
                z_dff[key] = pd.concat([
                    pd.DataFrame({'key': np.tile(key, len(y_data))}),
                    pd.DataFrame({'disp': y_data}),
                    zDict[year][key].iloc[:,2:]],
                                  axis=1)
            data = pd.concat([z_dff[key] for key in keys], axis=0).reset_index(drop=True)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

            for i in range(2):
                for j in range(2):
                    ax = axes[i * 2 + j]

                    subset = data.loc[data['disp'] == disp[i], :]
                    if subset.shape[0] > 1:
                        sns.kdeplot(
                            data=subset,
                            x=f'LatentFeature_{j}', ax=ax, legend=True,
                            hue='key',
                            multiple="stack",
                        )
                        ax.set_title(f'LatentFeature_{j}, MID = {disp[i]}', fontsize=13)
                    else:
                        ax.set_title(f'Skipped KDE: MID = {disp[i]} (N={subset.shape[0]})', fontsize=13)

                    ax.set_xlabel('')
                    ax.set_title(f'LatentFeature_{j}, MID = {disp[i]}', fontsize=13)

            suptitle = f'Year {year}, train: {count_of_ones["train"]} MIDs, '
            suptitle += f'test: {count_of_ones["test"]} MIDs \n'
            suptitle += f' Data ratios: training = 0.6, validation = 0.2, testing = 0.2'
            plt.suptitle(suptitle, fontsize=22)
            plt.tight_layout()

            pathfileh = self.folder + f'latentspace/pca_kde/'
            pathfileh += f'ARGVA_latent_space_PCA_features_kde_year_{year}.png'
            plt.savefig(pathfileh)
            plt.close()

        return

    def uploading_persistence_entropy(self):
        path0 = self.data_homology + f'Entropy_hom_H_012_{self.first_year}_{self.last_year}.csv'
        pe = pd.read_csv(path0, usecols = ['year','H_1','H_2']) 
        entropy = [pe.loc[pe['year']==year,:].values[:,1:]
                    for year in self.args['nyears']]
        entropy_values = [np.mean(e) for e in entropy]
        return entropy_values

    def pca_analysis(self):
        nyears = self.args['nyears']
        keys = ['train', 'test']
        explained_variances  = {'train': [], 'test': []}
        pairwise_distances = {'train': [], 'test': []}

        entropy_values = self.uploading_persistence_entropy()

        for key in keys:
            for year in nyears:
                dataloaders = self.get_data(year)
                model = self.loading_model(year)
                model.eval()

                dataloader = dataloaders[key]
                z_list, y_list = [], []
                for data in dataloader:
                    mask = getattr(data, f"{key}_mask").bool()
                    edge_index_masked, edge_attr_masked = self.masking_data(data, mask)

                    z = self.z_parameterize(model.encoder,
                                            data.x[mask],
                                            edge_index_masked,
                                            edge_attr_masked)
                    z_list.append(z.detach().cpu().numpy())
                    y_list.append(data.y[mask].detach().cpu().numpy().astype(np.int32))

                z_array = np.vstack(z_list)

                pca = PCA(n_components=2)
                transformed_data = pca.fit_transform(z_array)

                explained_variances[key].append(pca.explained_variance_ratio_)
                dist_matrix = distance.pdist(transformed_data)
                pairwise_distances[key].append(np.mean(dist_matrix))

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs = axs.flatten()
        linestyles = ['dashed', 'dotted']
        for i in range(2):
            for key in keys:
                variance = [v[i] for v in explained_variances[key]]
                axs[i].plot(nyears, variance, label=f'{key}', linestyle=linestyles[i])
            axs[i].set_title(f'Variance Explained by\n LatentFeature_{i}', fontsize = 18)
            axs[i].set_xlabel('Year')
            axs[i].legend()

        max_entropy = np.max(entropy_values)
        entropy_values = [max_entropy - e for e in entropy_values]
        for ax in axs:
            ax2 = ax.twinx()
            ax2.plot(nyears, entropy_values, color='blue', linestyle='-.', marker='+')
            ax2.set_ylabel('1 - Entropy', color='blue', fontsize = 14)
            ax2.tick_params(axis='y', labelcolor='blue')

        plt.tight_layout()

        pathfi = 'latentspace/ARGVA_latent_space_PCA_analysis.png'
        plt.savefig(self.folder + pathfi)
        plt.close()

        return

    def kmeans_cluster_latent_space(self):
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F833FF', '#33FFF8', '#F8FF33',
                  '#3F3F3F', '#FF5733']
        markers = [
            'o', # circle
            's', # square
            'D', # diamond
            '^', # triangle_up
            'v', # triangle_down
            '<', # triangle_left
            '>', # triangle_right
            '*', # star
        ]
        """ Kmeans """
        fig_kmeans, axes_kmeans = plt.subplots(2, 2, figsize=(12, 12))
        pca_results, all_disp = {}, {}

        for i, year in enumerate(self.args['specific_years']):
            model = self.loading_model(year)
            model.eval()

            dataloaders = self.get_data(year)
            dataloader = dataloaders['test']

            z_list, y_list = [], []
            for data in dataloader:
                mask = getattr(data, f"test_mask").bool()
                edge_index_masked, edge_attr_masked = self.masking_data(data, mask)

                z = self.z_parameterize(model.encoder,
                                        data.x[mask],
                                        edge_index_masked,
                                        edge_attr_masked)
                z_list.append(z.detach().cpu().numpy())
                y_list.append(data.y[mask].detach().cpu().numpy().astype(np.int32))

            z_array = np.vstack(z_list)
            all_disp[year] = np.hstack(y_list)

            pca = PCA(n_components=2)
            transformed_data = pca.fit_transform(z_array)
            reduced_data = transformed_data

            n_clusters = 5
            k_means = KMeans(init="k-means++", n_clusters=n_clusters, max_iter=2000,
                             random_state=0, n_init='auto')
            k_means.fit(reduced_data)

            k_means_cluster_centers = k_means.cluster_centers_
            k_means_labels = pairwise_distances_argmin(reduced_data,
                                                       k_means_cluster_centers)

            h = 0.001
            x_min, x_max = reduced_data[:, 0].min() - 1.5, reduced_data[:, 0].max() + 1.5
            y_min, y_max = reduced_data[:, 1].min() - 1.5, reduced_data[:, 1].max() + 1.5

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            grid_data = np.c_[xx.ravel(), yy.ravel()].astype(reduced_data.dtype)

            Z = k_means.predict(grid_data)
            Z = Z.reshape(xx.shape)

            colormap = plt.get_cmap('tab10')

            ax = axes_kmeans[i // 2, i % 2]
            ax.imshow(
                Z,
                interpolation="nearest",
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap=colormap,
                aspect="auto",
                origin="lower",
            )

            ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.",
                    label='Data Points', alpha=0.5)

            centroids = k_means.cluster_centers_
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker="*",
                s=100,
                linewidths=3,
                color="w",
                zorder=30,
                label='Centroids'
            )

            legend_patches = [patches.Patch(color='black', label='Data Points')]
            legend_patches.append(patches.Patch(color='white', label='Centroids'))
            ax.legend(handles=legend_patches, loc='center')

            x_min, x_max = centroids[:, 0].min() - 0.1, centroids[:, 0].max() + 0.1
            y_min, y_max = centroids[:, 1].min() - 0.1, centroids[:, 1].max() + 0.1

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())

            for j, centroid in enumerate(centroids):
                ax.annotate(
                    f'Centroid {j}',
                    xy=(centroid[0], centroid[1]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    ha='right',
                    va='bottom',
                    color="w",
                    fontsize=12,
                )

            ax.set_title( f"Year {year}", fontsize = 15)

        plt.suptitle(
            f"test: K-means clustering latent space \n centroids marked as white star",
            fontsize=18)
        plt.tight_layout()
        path_na = self.folder + 'test_Kmeans_clustering_four_years.png'
        plt.savefig(path_na)
        plt.close()

        for year in self.args['specific_years']:

            dataloaders = self.get_data(year)
            dataloader = dataloaders['test']

            z_list, y_list = [], []
            for data in dataloader:
                mask = getattr(data, f"test_mask").bool()
                edge_index_masked, edge_attr_masked = self.masking_data(data, mask)

                z = self.z_parameterize(model.encoder,
                                        data.x[mask],
                                        edge_index_masked,
                                        edge_attr_masked)
                z_list.append(z.detach().cpu().numpy())
                y_list.append(data.y[mask].detach().cpu().numpy().astype(np.int32))

            reduced_data = np.vstack(z_list)
            labels_true = np.hstack(y_list)

            n_clusters = 5

            mbk = MiniBatchKMeans(
                init="k-means++",
                n_clusters=n_clusters,
                batch_size=5,
                n_init=5,
                max_no_improvement=10,
                verbose=0,
            )
            mbk.fit(reduced_data)

            k_means = KMeans(init="k-means++", n_clusters=n_clusters, max_iter=2000,
                             random_state=0, n_init='auto')
            k_means.fit(reduced_data)

            k_means_cluster_centers = k_means.cluster_centers_
            order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
            mbk_means_cluster_centers = mbk.cluster_centers_[order]

            mbk_means_labels = pairwise_distances_argmin(reduced_data,
                                                         mbk_means_cluster_centers)

            """ MiniBatchKMeans """
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes = axes.flatten()

            for k, col in zip(range(n_clusters), colors):
                my_members = mbk_means_labels == k
                cluster_center = mbk_means_cluster_centers[k]
                axes[0].plot(reduced_data[my_members, 0],
                             reduced_data[my_members, 1],
                             "w", markerfacecolor=col, marker="*",
                             markersize=10)
                axes[0].plot(
                    cluster_center[0],
                    cluster_center[1],
                    "o",
                    markerfacecolor=col,
                    markeredgecolor="k",
                    markersize=10,
                )
            axes[0].set_title("test: MiniBatchKMeans")
            axes[0].set_xticks(())
            axes[0].set_yticks(())

            """ MeanShift """
            bandwidth = estimate_bandwidth(reduced_data, quantile=0.1, n_samples=1000)
            if bandwidth <= 0:
                bandwidth = 1e-3
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(reduced_data)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)

            for k, col in zip(range(n_clusters_), colors):
                my_members = labels == k
                cluster_center = cluster_centers[k]
                axes[1].plot(reduced_data[my_members, 0],
                             reduced_data[my_members, 1], markers[k], color=col)
                axes[1].plot(
                    cluster_center[0],
                    cluster_center[1],
                    markers[k],
                    markerfacecolor=col,
                    markeredgecolor="k",
                    markersize=20,
                 )
            axes[1].set_title("test: Meanshift \n Estimated number of clusters: %d" % n_clusters_)

            plt.tight_layout()
            path_name = f'test_Minibatches_Meanshift_plots_latentspace_year_{year}.png'
            plt.savefig(self.folder + path_name)
            plt.close()

        return

    def random_forest_classification(self):
        scores = {}
        for year in self.args['nyears']:

            model = self.loading_model(year)
            model.eval()

            dataloaders = self.get_data(year)
            dataloader = dataloaders['test']

            z_list, y_list = [], []
            for data in dataloader:
                mask = getattr(data, f"test_mask").bool()
                edge_index_masked, edge_attr_masked = self.masking_data(data, mask)

                z = self.z_parameterize(model.encoder,
                                        data.x[mask],
                                        edge_index_masked,
                                        edge_attr_masked)
                z_list.append(z.detach().cpu().numpy())
                y_list.append(data.y[mask].detach().cpu().numpy().astype(np.int32))

            z_array = np.vstack(z_list)
            labels_true = np.hstack(y_list)

            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(z_array)

            size = int(len(labels_true) / 5)
            clf = RandomForestClassifier(max_depth=20, n_estimators=200,
                                         random_state=10)
            clf.fit(reduced_data[:-size], labels_true[:-size])
            score = clf.score(np.array(reduced_data)[size:], labels_true[size:])
            scores[str(year)] =  np.mean(score)

        df_rsf_scores = pd.DataFrame(
            {'Year' : scores.keys(),
             'rsf_score': scores.values()
            })
        df_rsf_scores.to_csv(self.args['folder'] + f'latentspace_RSF_scores.csv',
                             float_format='%.3f',index = False)
        df_rsf_scores.plot(x='Year',y='rsf_score');
        plt.savefig(self.folder+'latentspace_rsf_scores.png')
        plt.close()

        return