
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import colormaps
import gudhi as gd
from gudhi.representations.metrics import pairwise_persistence_diagram_distances
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


class GudhiPlottingClass():

    def __init__(self,args):

        self.args = args
        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.n_comp = args['n_comp']
        self.cases = args['cases']
        self.years = args['years']
        self.vars = ['disp', 'majpower', 'allies', 'contgkf',
                       'smldep', 'lcaprat', 'lgmidist', 'smldmat']
        self.n_years = args['n_years']
        self.Homology = args['Homology']
        self.folder_images = self.Homology + 'images/'
        self.folder = args['folder_prep']

    """
    Uploading data
    """
    def data_loading(self):
        n_years = self.n_years
        Homology = self.Homology

        PDs = []
        for year in n_years:
            PD = pd.read_csv(Homology + f'Persistence_Diagrams/PD_hom_H_012_{year}.csv')
            PDs.append(PD)
        return PDs


    """
    Plotting diagrams
    """
    def Plotting_Persistence_Diagrams(self):
        chosen_years = self.years
        cases = self.cases
        PDs = self.data_loading()

        cmap = plt.get_cmap('tab10').colors

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()

        for i, idx in enumerate(cases):
            ax = axs[i]
            diagram_df = PDs[idx]
            
            # Convert to GUDHI format: list of (dim, (birth, death))
            diagram_list = [(int(row['h_dim']), (row['b'], row['d'])) for _, row in diagram_df.iterrows()]
            
            gd.plot_persistence_diagram(diagram_list, axes=ax,greyblock = False, colormap = cmap)
            ax.set_title(f"{chosen_years[i]}")
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.grid(False)

        plt.tight_layout()
        plt.savefig(self.folder_images + "Hom_Persistence_Diagrams_4_years.png")
        plt.close()
        return 


    """
    Plotting Pairwise Distances between Persistence Diagrams
    """
    def PDs_Pairwise_Distances(self):
        metric = 'wasserstein'
        PDs = self.data_loading()

        # Filter homology dimension 1 and extract [b, d]
        PDs_h1 = [df[df["h_dim"] == 1][["b", "d"]].to_numpy() for df in PDs]
        PWD = pairwise_persistence_diagram_distances(PDs_h1, metric=metric, n_jobs=3) 

        return PWD, metric


    def Plotting_PDs_Pairwise_Distances(self):
        n_years = self.n_years
        PWD, metric = self.PDs_Pairwise_Distances()

        fig, ax0 = plt.subplots(figsize=(12, 6))
        linkage_matrix = hierarchy.linkage(squareform(PWD), method='ward')

        labels = [str(year) for year in n_years]
        hierarchy.dendrogram(linkage_matrix, labels=labels, ax=ax0)
        ax0.set_xticklabels(ax0.get_xticklabels(), rotation=90, fontsize=6)
        ax0.set_title('Hierarchical Clustering Dendrogram')
        ax0.set_ylabel('Distance')
        plt.tight_layout()
        pathname = self.folder_images + f'Hom_PDs_{metric}_Dendrogram_Pairwise_Distances_all_years.png'
        plt.savefig(pathname)

        fig, ax1 = plt.subplots(figsize=(6, 6))
        cax = ax1.matshow(PWD, cmap='seismic')
        plt.colorbar(cax, shrink=0.75)

        ax1.set_xticks(np.arange(PWD.shape[0])[::5])
        ax1.set_yticks(np.arange(PWD.shape[0])[::5])
        ax1.set_xticklabels(n_years[::5], rotation=90, fontsize=10)
        ax1.set_yticklabels(n_years[::5], fontsize=10)
        ax1.xaxis.set_ticks_position('top')
        ax1.set_title(f"{metric} - Persistence-Diagrams Pairwise-Distance", fontsize=13)
        ax1.set_facecolor("white")
        ax1.grid(False)

        plt.tight_layout()
        pathheat = self.folder_images + f'Hom_PDs_{metric}_Heatmap_Pairwise_Distances_all_years.png'
        plt.savefig(pathheat)
        plt.close()
        return


    def Saving_Dendrogram(self):
        n_years = self.n_years
        PWD, metric = self.PDs_Pairwise_Distances()
        
        Z = hierarchy.linkage(squareform(PWD), method='ward')
        labels = [str(year) for year in n_years]

        def get_all_level_clusters(Z, n, labels):
            all_clusters = {}
            for level in range(1, n + 1):
                clusters_at_level = hierarchy.fcluster(Z, level, criterion='maxclust')
                all_clusters[level] = {i: [] for i in range(1, level + 1)}
                for label_idx, cluster in enumerate(clusters_at_level):
                    all_clusters[level][cluster].append(labels[label_idx])
            return all_clusters

        n = PWD.shape[0]
        all_level_clusters = get_all_level_clusters(Z, n, labels)

        def create_cluster_df(all_level_clusters):
            df_data = []
            for level, clusters in all_level_clusters.items():
                for cluster_id, members in clusters.items():
                    df_data.append({
                        "Level": level,
                        "Cluster": cluster_id,
                        "Labels": ', '.join(members)
                    })
            return pd.DataFrame(df_data)

        cluster_df_level_index = create_cluster_df(all_level_clusters)
        filelc = self.folder_images + f'Hom_PDs_{metric}_Hierarchical_Clustering_Dendrogram_data.csv'
        cluster_df_level_index.to_csv(filelc)

        return
    

    def silhouette_score(self):

        PWD,metric = self.PDs_Pairwise_Distances()

        n_years = self.n_years
        cluster_range = range(2, len(n_years))

        silhouette_scores = []
        for num_clusters in cluster_range:
            kmeans = KMeans(n_clusters=num_clusters, n_init = 'auto', random_state=42)
            cluster_labels = kmeans.fit_predict(PWD)
            silhouette_avg = silhouette_score(PWD, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        plt.figure(figsize=(8, 6))
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.savefig(self.folder_images + f'Hom_PDs_{metric}_Silhouette_Scores_all_years.png');
        plt.close()
        return

    """
    Plotting barcodes
    """

    def Plotting_Persistence_Barcodes(self):
        cases = self.cases
        chosen_years = self.years
        PDs = self.data_loading()
        pds = [PDs[i] for i in cases]
        cmap = plt.get_cmap('tab10').colors

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for year, ax, diag_df in zip(chosen_years, axes, pds):
            # Convert to list of (h_dim, (birth, death))
            diag_tuples = [(int(row['h_dim']), (row['b'], row['d']))
               for _, row in diag_df.iterrows() if int(row['h_dim']) != 0]
            gd.plot_persistence_barcode(diag_tuples, axes=ax, colormap = cmap)
            ax.set_title(f'{year}', fontsize=14)
            ax.grid(False)

        plt.suptitle('Persistence Barcodes', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.folder_images + 'Hom_Persistence_Barcodes_4_years.png')
        plt.close()

        return 


    def Plotting_Persistence_Entropy(self):
        df = pd.read_csv(self.Homology + f'Entropy_hom_H_012_{self.beginYear}_{self.endYear}.csv')

        labels = {0: r'$H_0$', 
                  1: r'$H_1$', 
                  2: r'$H_2$'}

        fig = plt.figure(figsize=(10, 7))

        for i, key in zip([0,1,2], ['H_0', 'H_1', 'H_2']):
            
            plt.plot(df['year'], df[key], label=labels[i])
            plt.scatter(df['year'], df[key])

        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Entropy', fontsize=14)
        plt.title('Persistence Entropy by Homology Dimension', fontsize=18)
        plt.legend()

        plt.grid(False)
        plt.savefig(self.folder_images + "Hom_Persistence_Entropy_all_years.png")
        plt.close()

