
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import decomposition

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
from scipy.cluster import hierarchy
from scipy.interpolate import UnivariateSpline

import gudhi as gd
from gudhi.representations.vector_methods import Entropy

import scipy.stats as stats
import xarray as xr

class GudhiPlottingPropertiesClass:

    def __init__(self, args):
        self.args = args
        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.n_years = args['n_years']
        self.homology_dimensions = args['homology_dimensions']
        self.data_prep = args['folder_prep']
        self.Homology = args['Homology']
        self.n_components = args['n_components']

        self.entropy_data = self.persistence_entropy()
        self.fpca_data = self.fpca()
        self.compute_silhouette_scores = self.compute_silhouette_scores()
        
        self.yi = xr.open_dataset(self.data_prep + 'yi.nc').to_dataframe()
        self.xij = xr.open_dataset(self.data_prep + 'xij.nc').to_dataframe()
        self.IOhom = pd.read_csv(self.Homology + f'IO_data_hom_H_012_{self.beginYear}_{self.endYear}.csv',
                                usecols = ['year'] + [f'H_{i}' for i in range(1,self.n_components + 1)])

    def data_loading(self):
        PDs = []
        for year in self.n_years:
            PD = pd.read_csv( self.Homology + f'Persistence_Diagrams/PD_hom_H_012_{year}.csv')
            PDs.append(PD)
            
        PDs =  [df[df['h_dim'] == 1][['b', 'd']].to_numpy() for df in PDs]

        return PDs

    def smooth_with_splines(self, x, y, s=6):
        """
        Smooths out a vector of values using splines.
        Parameters:
        x (numpy.ndarray): The x-coordinates of the data points.
        y (numpy.ndarray): The y-coordinates of the data points.
        s (float): Positive smoothing factor used to choose the number of knots.
                Default is 1. Larger values mean more smoothing.
        Returns:
        numpy.ndarray: The smoothed y-values.
        """
        spline = UnivariateSpline(x, y, s=s)
        y_smooth = spline(x)
        return y_smooth

    def persistence_entropy(self):
        PDs = self.data_loading()
        PE = Entropy()
        entropy = np.array(PE.fit_transform(PDs))
        entropy /=np.max(entropy)
        return entropy 

    def fpca(self):
        return pd.read_csv(self.Homology +
                    f'FPCA_hom_H_012_{self.beginYear}_{self.endYear}.csv' )


    def compute_silhouette_scores(self):

        entropy_data = self.entropy_data
        entropy_data = np.concatenate(entropy_data, axis=0).reshape(-1,1)

        n_years = self.n_years
        cluster_range = range(2, len(n_years) // 2 + 1)

        silhouette_scores = []
        silhouette_samples_scores_all = {}
        cluster_labels_all = {}

        for num_clusters in cluster_range:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init = 'auto')
            cluster_labels = kmeans.fit_predict(entropy_data)
            silhouette_avg = silhouette_score(entropy_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            silhouette_samples_scores_all[num_clusters] = silhouette_samples(entropy_data,
                                                                             cluster_labels)
            cluster_labels_all[num_clusters] = cluster_labels

        return {
            'silhouette_scores': silhouette_scores,
            'silhouette_samples_scores_all': silhouette_samples_scores_all,
            'cluster_labels_all': cluster_labels_all,
            'cluster_range': cluster_range,
        }

    def silhouette_scores(self):
        result = self.compute_silhouette_scores

        silhouette_scores  = result['silhouette_scores']
        cluster_range = result['cluster_range']

        plt.figure(figsize=(15, 10))
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='solid',
                 linewidth=2, markersize=8)
        plt.title('Silhouette Score vs. Number of Clusters', fontsize=22)
        plt.xlabel('Number of Clusters', fontsize=18)
        plt.ylabel('Silhouette Score', fontsize=18)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(self.Homology + f'images/Hom_PDs_Silhouette_Scores_all_years.png')
        plt.close()
        return

    def silhouette_score_per_cluster_over_years(self):
        result = self.compute_silhouette_scores
        silhouette_samples_scores_all = result['silhouette_samples_scores_all']
        cluster_range = result['cluster_range']

        n_years = self.n_years

        silhouette_scores = {k: [] for k in cluster_range}

        for num_clusters in cluster_range:
            silhouette_samples_scores = silhouette_samples_scores_all[num_clusters]
            for year, score in zip(n_years, silhouette_samples_scores):
                silhouette_scores[num_clusters].append((year, score))

        # Transforming data for plotting
        plot_data = {}
        for num_clusters in cluster_range:
            for year, score in silhouette_scores[num_clusters]:
                if year not in plot_data:
                    plot_data[year] = {}
                plot_data[year][num_clusters] = score

        # Creating the bar plot
        fig, ax = plt.subplots(figsize=(15, 10))

        bar_width = 0.75
        bar_positions = np.arange(len(n_years))

        bottom = np.zeros(len(n_years))
        for num_clusters in cluster_range:
            heights = [plot_data[year].get(num_clusters, 0) for year in n_years]
            ax.bar(bar_positions, heights, bar_width, bottom=bottom,
                   label=f'{num_clusters} Clusters')
            bottom += heights

        ax.set_xticks(bar_positions)
        ax.set_xticklabels(n_years, rotation=45, fontsize=14)
        ax.set_xlabel('Year', fontsize=18)
        ax.set_ylabel('Silhouette Score', fontsize=18)
        ax.set_title('Silhouette Scores by Year and Number of Clusters', fontsize=22)
        ax.legend(title='Number of Clusters', loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        pathw = self.Homology + 'images/Hom_PDs_Silhouette_Scores_by_Year_and_Number_of_Clusters.png'
        plt.savefig(pathw)
        plt.close()
        return

    def average_silhouette_score_and_total_mids_over_years(self):
        yi = self.yi
        n_years = self.n_years

        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()}
                    for year in n_years}

        result = self.compute_silhouette_scores
        silhouette_samples_scores_all = result['silhouette_samples_scores_all']

        max_scores = []
        min_scores = []
        avg_scores = []

        for year_index in range(len(n_years)):
            scores_for_year = []
            for num_clusters in silhouette_samples_scores_all:
                silhouette_scores = silhouette_samples_scores_all[num_clusters]
                scores_for_year.append(silhouette_scores[year_index])
            max_scores.append(np.max(scores_for_year))
            min_scores.append(np.min(scores_for_year))
            avg_scores.append(np.mean(scores_for_year))
        mids = [midsDict[year]['mids'] for year in n_years]

        fig, ax1 = plt.subplots(figsize=(15, 10))
        color = 'tab:blue'
        ax1.set_xlabel('Year', fontsize=20)
        ax1.set_ylabel('Scores', color=color, fontsize=20)
        ax1.plot(n_years, avg_scores, marker='^', linestyle='solid',
                 linewidth=3, markersize=10, color='tab:blue', label='Average Score')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Silhouette Scores and Total MIDs over Years', fontsize=22)
        ax1.grid(axis = 'x')

        ax1.axvline(x=1991, color='gray', linestyle='--', linewidth=2)

        ax1.annotate('End of Cold War', xy=(1991, min(avg_scores)), xycoords='data',
                 xytext=(2000, min(avg_scores)), textcoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 horizontalalignment='left', verticalalignment='top', fontsize=15)


        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Total MIDs', color=color, fontsize=20)
        ax2.plot(n_years, mids, marker='d', linestyle='solid', linewidth=3,
                 markersize=10, color=color, label='Total MIDs')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(axis = 'x')
        fig.legend(loc='upper left', bbox_to_anchor=(0.25, 0.85), fontsize=15)
        pathw = self.Homology + 'images/Hom_PDs_Silhouette_Scores_and_MIDs.png'
        plt.savefig(pathw)
        plt.close()

        return

    def average_silhouette_score_vs_variables_over_years(self):

        yi = self.yi
        xij = self.xij
        n_years = self.n_years

        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()} for year in n_years}
        alliesDict = {year: {'allies': xij.loc[xij['year'] == year, 'allies'].sum()} for year in n_years}
        lcapratDict = {year: {'lcaprat': xij.loc[xij['year'] == year, 'lcaprat'].sum()} for year in n_years}
        smldepDict = {year: {'smldep': xij.loc[xij['year'] == year, 'smldep'].sum()} for year in n_years}
        lgmidistDict = {year: {'lgmidist': xij.loc[xij['year'] == year, 'lgmidist'].sum()} for year in n_years}
        smldmatDict = {year: {'smldmat': xij.loc[xij['year'] == year, 'smldmat'].sum()} for year in n_years}

        result = self.compute_silhouette_scores
        silhouette_samples_scores_all = result['silhouette_samples_scores_all']

        avg_scores = []

        for year_index in range(len(n_years)):
            scores_for_year = []
            for num_clusters in silhouette_samples_scores_all:
                silhouette_scores = silhouette_samples_scores_all[num_clusters]
                scores_for_year.append(silhouette_scores[year_index])
            avg_scores.append(np.mean(scores_for_year))

        mids = [midsDict[year]['mids'] for year in n_years]
        allies = [alliesDict[year]['allies'] for year in n_years]
        lcaprat = [lcapratDict[year]['lcaprat'] for year in n_years]
        smldep = [smldepDict[year]['smldep'] for year in n_years]
        lgmidist = [lgmidistDict[year]['lgmidist'] for year in n_years]
        smldmat = [smldmatDict[year]['smldmat'] for year in n_years]

        fig, axs = plt.subplots(2, 3, figsize=(30, 20))

        # Plot MIDs and Silhouette Scores
        color = 'tab:blue'
        axs[0, 0].set_xlabel('Year', fontsize=18)
        axs[0, 0].set_ylabel('Average Silhouette Score', color=color, fontsize=18)
        axs[0, 0].plot(n_years, avg_scores, marker='^', linestyle='solid', linewidth=3, markersize=10,
                       color='tab:blue',
                       label='Average Silhouette Score')
        axs[0, 0].tick_params(axis='y', labelcolor=color)
        axs[0, 0].set_title('Average Silhouette Score and MIDs over Years', fontsize=22)

        ax1_twin = axs[0, 0].twinx()
        color = 'tab:orange'
        ax1_twin.set_ylabel('Number of MIDs', color=color, fontsize=18)
        ax1_twin.plot(n_years, mids, marker='d', linestyle='solid', linewidth=3, markersize=10,
                      color=color, label='MIDs')
        ax1_twin.tick_params(axis='y', labelcolor=color)

        axs[0, 0].legend(loc='upper left', fontsize=14)
        ax1_twin.legend(loc='upper right', fontsize=14)
        axs[0, 0].grid(True)

        # Plot Allies and Silhouette Scores
        color = 'tab:blue'
        axs[0, 1].set_xlabel('Year', fontsize=18)
        axs[0, 1].set_ylabel('Average Silhouette Score', color=color, fontsize=18)
        axs[0, 1].plot(n_years, avg_scores, marker='^', linestyle='solid', linewidth=3, markersize=10,
                       color='tab:blue',
                        label='Average Silhouette Score')
        axs[0, 1].tick_params(axis='y', labelcolor=color)
        axs[0, 1].set_title('Average Silhouette Score and Allies over Years', fontsize=22)

        ax2_twin = axs[0, 1].twinx()
        color = 'tab:green'
        ax2_twin.set_ylabel('Sum of Dyadic-Allies', color=color, fontsize=18)
        ax2_twin.plot(n_years, allies, marker='s', linestyle='solid', linewidth=3, markersize=10,
                      color=color, label='Allies')
        ax2_twin.tick_params(axis='y', labelcolor=color)

        axs[0, 1].legend(loc='upper left', fontsize=14)
        ax2_twin.legend(loc='upper right', fontsize=14)
        axs[0, 1].grid(True)

        # Plot Lcaprat and Silhouette Scores
        color = 'tab:blue'
        axs[0, 2].set_xlabel('Year', fontsize=18)
        axs[0, 2].set_ylabel('Average Silhouette Score', color=color, fontsize=18)
        axs[0, 2].plot(n_years, avg_scores, marker='^', linestyle='solid', linewidth=3, markersize=10,
                       color='tab:blue',
                       label='Average Silhouette Score')
        axs[0, 2].tick_params(axis='y', labelcolor=color)
        axs[0, 2].set_title('Average Silhouette Score and Lcaprat over Years', fontsize=22)

        ax3_twin = axs[0, 2].twinx()
        color = 'tab:red'
        ax3_twin.set_ylabel('Sum of Lcaprat', color=color, fontsize=18)
        ax3_twin.plot(n_years, lcaprat, marker='^', linestyle='solid', linewidth=3, markersize=10,
                      color=color,
                      label='Lcaprat')
        ax3_twin.tick_params(axis='y', labelcolor=color)

        axs[0, 2].legend(loc='upper left', fontsize=14)
        ax3_twin.legend(loc='upper right', fontsize=14)
        axs[0, 2].grid(True)

        # Plot Smldep and Silhouette Scores
        color = 'tab:blue'
        axs[1, 0].set_xlabel('Year', fontsize=18)
        axs[1, 0].set_ylabel('Average Silhouette Score', color=color, fontsize=18)
        axs[1, 0].plot(n_years, avg_scores, marker='^', linestyle='solid', linewidth=3, markersize=10,
                       color='tab:blue',
                       label='Average Silhouette Score')
        axs[1, 0].tick_params(axis='y', labelcolor=color)
        axs[1, 0].set_title('Average Silhouette Score and SMLDEP over Years', fontsize=22)

        ax4_twin = axs[1, 0].twinx()
        color = 'tab:purple'
        ax4_twin.set_ylabel('Sum of SMLDEP', color=color, fontsize=18)
        ax4_twin.plot(n_years, smldep, marker='v', linestyle='solid', linewidth=3, markersize=10, color=color, label='SMLDEP')
        ax4_twin.tick_params(axis='y', labelcolor=color)

        axs[1, 0].legend(loc='upper left', fontsize=14)
        ax4_twin.legend(loc='upper right', fontsize=14)
        axs[1, 0].grid(True)

        # Plot Lgmidist and Silhouette Scores
        color = 'tab:blue'
        axs[1, 1].set_xlabel('Year', fontsize=18)
        axs[1, 1].set_ylabel('Average Silhouette Score', color=color, fontsize=18)
        axs[1, 1].plot(n_years, avg_scores, marker='^', linestyle='solid', linewidth=3, markersize=10,
                       color='tab:blue',
                       label='Average Silhouette Score')
        axs[1, 1].tick_params(axis='y', labelcolor=color)
        axs[1, 1].set_title('Average Silhouette Score and LGMIDIST over Years', fontsize=22)

        ax5_twin = axs[1, 1].twinx()
        color = 'tab:brown'
        ax5_twin.set_ylabel('Sum of LGMIDIST', color=color, fontsize=18)
        ax5_twin.plot(n_years, lgmidist, marker='p', linestyle='solid', linewidth=3, markersize=10,
                      color=color,
                      label='LGMIDIST')
        ax5_twin.tick_params(axis='y', labelcolor=color)

        axs[1, 1].legend(loc='upper left', fontsize=14)
        ax5_twin.legend(loc='upper right', fontsize=14)
        axs[1, 1].grid(True)

        # Plot Smldmat and Silhouette Scores
        color = 'tab:blue'
        axs[1, 2].set_xlabel('Year', fontsize=18)
        axs[1, 2].set_ylabel('Average Silhouette Score', color=color, fontsize=18)
        axs[1, 2].plot(n_years, avg_scores, marker='^', linestyle='solid', linewidth=3, markersize=10,
                       color='tab:blue',
                       label='Average Silhouette Score')
        axs[1, 2].tick_params(axis='y', labelcolor=color)
        axs[1, 2].set_title('Average Silhouette Score and SMLDMAT over Years', fontsize=22)

        ax6_twin = axs[1, 2].twinx()
        color = 'tab:cyan'
        ax6_twin.set_ylabel('Sum of SMLDMAT', color=color, fontsize=18)
        ax6_twin.plot(n_years, smldmat, marker='h', linestyle='solid', linewidth=3, markersize=10, color=color,
                      label='SMLDMAT')
        ax6_twin.tick_params(axis='y', labelcolor=color)

        axs[1, 2].legend(loc='upper left', fontsize=14)
        ax6_twin.legend(loc='upper right', fontsize=14)
        axs[1, 2].grid(True)

        fig.tight_layout()
        pathw = self.Homology + 'images/Hom_PDs_Avg_Silhouette_Scores_Variables_over_Years.png'
        plt.savefig(pathw)
        plt.close()

        return

    def plot_entropy_mids_allies_and_lcaprat_over_years(self):

        yi = self.yi
        xij = self.xij

        n_years = self.n_years

        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()}
                    for year in n_years}

        alliesDict = {year: {'allies': xij.loc[yi['year'] == year, 'allies'].sum()}
                    for year in n_years}

        lcapratDict = {year: {'lcaprat': xij.loc[yi['year'] == year, 'lcaprat'].sum()}
                       for year in n_years}

        smldepDict = {year: {'smldep': xij.loc[xij['year'] == year, 'smldep'].sum()}
                    for year in n_years}
        lgmidistDict = {year: {'lgmidist': xij.loc[xij['year'] == year, 'lgmidist'].sum()}
                    for year in n_years}
        smldmatDict = {year: {'smldmat': xij.loc[xij['year'] == year, 'smldmat'].sum()}
                    for year in n_years}

        entropy_values =  self.entropy_data
        mids = [midsDict[year]['mids'] for year in n_years]
        allies = [alliesDict[year]['allies'] for year in n_years]
        lcaprat = [lcapratDict[year]['lcaprat'] for year in n_years]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        # Plot MIDs and Entropy
        color = 'tab:blue'
        ax1.set_xlabel('Year', fontsize=18)
        ax1.set_ylabel('Persistence Entropy', color=color, fontsize=18)
        ax1.plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                 label='Persistence Entropy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Persistence Entropy and MIDs over Years', fontsize=22)

        ax1_twin = ax1.twinx()
        color = 'tab:orange'
        ax1_twin.set_ylabel('Sum of MIDs', color=color, fontsize=18)
        ax1_twin.plot(n_years, mids, marker='d', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='MIDs')
        ax1_twin.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        ax1.legend(loc='upper left', fontsize=14)
        ax1_twin.legend(loc='upper right', fontsize=14)
        ax1.grid(True)

        # Plot Allies and Entropy
        color = 'tab:blue'
        ax2.set_xlabel('Year', fontsize=18)
        ax2.set_ylabel('Persistence Entropy', color=color, fontsize=18)
        ax2.plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                 label='Persistence Entropy')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_title('Persistence Entropy and Allies over Years', fontsize=22)

        ax2_twin = ax2.twinx()
        color = 'tab:green'
        ax2_twin.set_ylabel('Sum of Dyadic-Allies', color=color, fontsize=18)
        ax2_twin.plot(n_years, allies, marker='s', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='Allies')
        ax2_twin.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        ax2.legend(loc='upper left', fontsize=14)
        ax2_twin.legend(loc='upper right', fontsize=14)
        ax2.grid(True)

        # Plot lcaprat and Entropy
        color = 'tab:blue'
        ax3.set_xlabel('Year', fontsize=18)
        ax3.set_ylabel('Persistence Entropy', color=color, fontsize=18)
        ax3.plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                 label='Persistence Entropy')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_title('Persistence Entropy and Lcaprat over Years', fontsize=22)

        ax3_twin = ax3.twinx()
        color = 'tab:red'
        ax3_twin.set_ylabel('Sum of Lcaprat', color=color, fontsize=18)
        ax3_twin.plot(n_years, lcaprat, marker='^', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='Lcaprat')
        ax3_twin.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        ax3.legend(loc='upper left', fontsize=14)
        ax3_twin.legend(loc='upper right', fontsize=14)
        ax3.grid(True)

        pathw = self.Homology + 'images/Hom_PDs_Entropy_MIDs_Allies_Lcaprat_over_Years.png'
        plt.savefig(pathw)
        plt.close()

        return

    def smooth_multivariate_with_splines(self, x, y_dict, s=10):
        y_smooth_dict = {}
        for feature, values in y_dict.items():
            values = np.array(values)
            spline = UnivariateSpline(x, values, s=s,k=5)
            y_smooth_dict[feature] = spline(x)
        return y_smooth_dict


    def entropy_and_covariates_over_years(self):

        yi = self.yi
        xij =self.xij
        n_years = self.n_years

        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()}
                    for year in n_years}

        alliesDict = {year: {'allies': xij.loc[xij['year'] == year, 'allies'].sum()}
                    for year in n_years}

        lcapratDict = {year: {'lcaprat': xij.loc[xij['year'] == year, 'lcaprat'].sum()}
                       for year in n_years}

        smldepDict = {year: {'smldep': xij.loc[xij['year'] == year, 'smldep'].sum()}
                    for year in n_years}

        lgmidistDict = {year: {'lgmidist': xij.loc[xij['year'] == year, 'lgmidist'].sum()}
                    for year in n_years}

        smldmatDict = {year: {'smldmat': xij.loc[xij['year'] == year, 'smldmat'].sum()}
                    for year in n_years}

        entropy_values = self.entropy_data
        mids = [midsDict[year]['mids'] for year in n_years]
        allies = [alliesDict[year]['allies'] for year in n_years]
        lcaprat = [lcapratDict[year]['lcaprat'] for year in n_years]
        smldep = [smldepDict[year]['smldep'] for year in n_years]
        lgmidist = [lgmidistDict[year]['lgmidist'] for year in n_years]
        smldmat = [smldmatDict[year]['smldmat'] for year in n_years]

        fig, axs = plt.subplots(2, 3, figsize=(30, 20))

        # Plot MIDs and Entropy
        color = 'tab:blue'
        axs[0, 0].set_xlabel('Year', fontsize=18)
        axs[0, 0].set_ylabel('Persistence Entropy', color=color, fontsize=18)
        axs[0, 0].plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                       label='Persistence Entropy')
        axs[0, 0].tick_params(axis='y', labelcolor=color)
        axs[0, 0].set_title('Persistence Entropy and MIDs over Years', fontsize=22)

        ax1_twin = axs[0, 0].twinx()
        color = 'tab:orange'
        ax1_twin.set_ylabel('Sum of MIDs', color=color, fontsize=18)
        ax1_twin.plot(n_years, mids, marker='d', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='MIDs')
        ax1_twin.tick_params(axis='y', labelcolor=color)

        axs[0, 0].legend(loc='upper left', fontsize=14)
        ax1_twin.legend(loc='upper right', fontsize=14)
        axs[0, 0].grid(True)

        # Plot Allies and Entropy
        color = 'tab:blue'
        axs[0, 1].set_xlabel('Year', fontsize=18)
        axs[0, 1].set_ylabel('Persistence Entropy', color=color, fontsize=18)
        axs[0, 1].plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                       label='Persistence Entropy')
        axs[0, 1].tick_params(axis='y', labelcolor=color)
        axs[0, 1].set_title('Persistence Entropy and Allies over Years', fontsize=22)

        ax2_twin = axs[0, 1].twinx()
        color = 'tab:green'
        ax2_twin.set_ylabel('Sum of Dyadic-Allies', color=color, fontsize=18)
        ax2_twin.plot(n_years, allies, marker='s', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='Allies')
        ax2_twin.tick_params(axis='y', labelcolor=color)

        axs[0, 1].legend(loc='upper left', fontsize=14)
        ax2_twin.legend(loc='upper right', fontsize=14)
        axs[0, 1].grid(True)

        # Plot lcaprat and Entropy
        color = 'tab:blue'
        axs[0, 2].set_xlabel('Year', fontsize=18)
        axs[0, 2].set_ylabel('Persistence Entropy', color=color, fontsize=18)
        axs[0, 2].plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                       label='Persistence Entropy')
        axs[0, 2].tick_params(axis='y', labelcolor=color)
        axs[0, 2].set_title('Persistence Entropy and Lcaprat over Years', fontsize=22)

        ax3_twin = axs[0, 2].twinx()
        color = 'tab:red'
        ax3_twin.set_ylabel('Sum of Lcaprat', color=color, fontsize=18)
        ax3_twin.plot(n_years, lcaprat, marker='^', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='Lcaprat')
        ax3_twin.tick_params(axis='y', labelcolor=color)

        axs[0, 2].legend(loc='upper left', fontsize=14)
        ax3_twin.legend(loc='upper right', fontsize=14)
        axs[0, 2].grid(True)

        # Plot smldep and Entropy
        color = 'tab:blue'
        axs[1, 0].set_xlabel('Year', fontsize=18)
        axs[1, 0].set_ylabel('Persistence Entropy', color=color, fontsize=18)
        axs[1, 0].plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                       label='Persistence Entropy')
        axs[1, 0].tick_params(axis='y', labelcolor=color)
        axs[1, 0].set_title('Persistence Entropy and SMLDEP over Years', fontsize=22)

        ax4_twin = axs[1, 0].twinx()
        color = 'tab:purple'
        ax4_twin.set_ylabel('Sum of SMLDEP', color=color, fontsize=18)
        ax4_twin.plot(n_years, smldep, marker='v', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='SMLDEP')
        ax4_twin.tick_params(axis='y', labelcolor=color)

        axs[1, 0].legend(loc='upper left', fontsize=14)
        ax4_twin.legend(loc='upper right', fontsize=14)
        axs[1, 0].grid(True)

        # Plot lgmidist and Entropy
        color = 'tab:blue'
        axs[1, 1].set_xlabel('Year', fontsize=18)
        axs[1, 1].set_ylabel('Persistence Entropy', color=color, fontsize=18)
        axs[1, 1].plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                       label='Persistence Entropy')
        axs[1, 1].tick_params(axis='y', labelcolor=color)
        axs[1, 1].set_title('Persistence Entropy and LGMIDIST over Years', fontsize=22)

        ax5_twin = axs[1, 1].twinx()
        color = 'tab:brown'
        ax5_twin.set_ylabel('Sum of LGMIDIST', color=color, fontsize=18)
        ax5_twin.plot(n_years, lgmidist, marker='p', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='LGMIDIST')
        ax5_twin.tick_params(axis='y', labelcolor=color)

        axs[1, 1].legend(loc='upper left', fontsize=14)
        ax5_twin.legend(loc='upper right', fontsize=14)
        axs[1, 1].grid(True)

        # Plot smldmat and Entropy
        color = 'tab:blue'
        axs[1, 2].set_xlabel('Year', fontsize=18)
        axs[1, 2].set_ylabel('Persistence Entropy', color=color, fontsize=18)
        axs[1, 2].plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=2,
                       label='Persistence Entropy')
        axs[1, 2].tick_params(axis='y', labelcolor=color)
        axs[1, 2].set_title('Persistence Entropy and SMLDMAT over Years', fontsize=22)

        ax6_twin = axs[1, 2].twinx()
        color = 'tab:cyan'
        ax6_twin.set_ylabel('Sum of SMLDMAT', color=color, fontsize=18)
        ax6_twin.plot(n_years, smldmat, marker='h', linestyle='-', linewidth=2,
                      markersize=10, color=color, label='SMLDMAT')
        ax6_twin.tick_params(axis='y', labelcolor=color)

        axs[1, 2].legend(loc='upper left', fontsize=14)
        ax6_twin.legend(loc='upper right', fontsize=14)
        axs[1, 2].grid(True)

        fig.tight_layout()
        pathw = self.Homology + 'images/Hom_PDs_splines_Entropy_MIDs_Allies_Lcaprat_SMLDEP'
        pathw += '_LGMIDIST_SMLDMAT_over_Years.png'
        plt.savefig(pathw)
        plt.close()

        return
    
    def plot_mids_vs_silhouette_scores(self):

        yi = self.yi
        n_years = self.n_years

        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()}
                    for year in n_years}

        result = self.compute_silhouette_scores
        silhouette_samples_scores_all = result['silhouette_samples_scores_all']

        avg_scores = []

        for year_index in range(len(n_years)):
            scores_for_year = []
            for num_clusters in silhouette_samples_scores_all:
                silhouette_scores = silhouette_samples_scores_all[num_clusters]
                scores_for_year.append(silhouette_scores[year_index])
            avg_scores.append(np.mean(scores_for_year))

        mids = [midsDict[year]['mids'] for year in n_years]

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(avg_scores, mids, color='tab:blue', s=100)
        ax.set_xlabel('Average Silhouette Score', fontsize=18)
        ax.set_ylabel('Total MIDs', fontsize=18)
        ax.set_title('Total MIDs vs. Average Silhouette Scores', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis='x')

        pathw = self.Homology + 'images/Hom_PDs_MIDs_vs_Silhouette_Scores.png'
        plt.savefig(pathw)
        plt.close()

        return

    def plot_entropy_and_mids_over_years(self):

        yi = self.yi
        n_years = self.n_years

        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()}
                    for year in n_years}

        entropy_values =  self.entropy_data
        mids = [midsDict[year]['mids'] for year in n_years]

        fig, ax1 = plt.subplots(figsize=(15, 10))

        color = 'tab:blue'
        ax1.set_xlabel('Year', fontsize=20)
        ax1.set_ylabel('Persistence Entropy', color=color, fontsize=20)
        ax1.plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=3,
                 label='Persistence Entropy')
        ax1.tick_params(axis='y', labelcolor=color, size = 10)
        ax1.set_title('Persistence Entropy and Total MIDs over Years', fontsize=22)
        ax1.grid(axis = 'x')

        ax1.axvline(x=1991, color='gray', linestyle='--', linewidth=2)

        ax1.annotate('End of Cold War', xy=(1991, min(entropy_values)), xycoords='data',
                     xytext=(2000, min(entropy_values)), textcoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                     horizontalalignment='left', verticalalignment='top', fontsize=15)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Total MIDs', color=color, fontsize=20)
        ax2.plot(n_years, mids, marker='d', linestyle='-', linewidth=3,
                 markersize=10, color=color, label='MIDs')
        ax2.tick_params(axis='y', labelcolor=color, size = 10)
        ax2.grid(axis = 'x')

        fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85), fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        pathw = self.Homology + 'images/Hom_PDs_Entropy_and_MIDs_over_Years.png'
        plt.savefig(pathw)
        plt.close()

        return

    def plot_entropy_and_mids_fpca_over_years(self):

        yi = self.yi
        n_years = self.n_years

        fp = self.fpca().values
        
        fpca = np.mean(fp,axis=1)/np.max(fp) 

        midsDict = {year: {'mids': yi.loc[yi['year'] == year, 'disp'].sum()}
                    for year in n_years}

        entropy_values = self.entropy_data
        mids = [midsDict[year]['mids'] for year in n_years]

        fig, ax1 = plt.subplots(figsize=(15, 10))

        color = 'tab:blue'
        ax1.set_xlabel('Year', fontsize=20)
        ax1.set_ylabel('Persistence Entropy', color=color, fontsize=20)
        ax1.plot(n_years, entropy_values, marker='o', linestyle='-', color=color, linewidth=3,
                label='Persistence Entropy')
        ax1.tick_params(axis='y', labelcolor=color, size=10)
        ax1.set_title('Persistence Entropy, Total MIDs, and FPCA over Years', fontsize=22)
        ax1.grid(axis='x')

        # Add the second y-axis for Total MIDs
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Total MIDs', color=color, fontsize=20)
        ax2.plot(n_years, mids, marker='d', linestyle='-', linewidth=3,
                markersize=10, color=color, label='MIDs')
        ax2.tick_params(axis='y', labelcolor=color, size=10)
        ax2.spines['right'].set_position(('outward', 60))

        # Add the third y-axis for Mean FPCA
        ax3 = ax1.twinx()
        color = 'tab:green'
        ax3.set_ylabel('Normalized Mean FPCA', color=color, fontsize=20)
        ax3.plot(n_years, fpca, marker='s', linestyle='-', linewidth=3,
                markersize=10, color=color, label='FPCA')
        ax3.tick_params(axis='y', labelcolor=color, size=10)
        ax3.spines['right'].set_position(('outward', 120))

        fig.legend(loc='upper left', bbox_to_anchor=(0.05, 0.65), fontsize=13)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        fig.tight_layout()

        pathw = self.Homology + 'images/Hom_PDs_Entropy_MIDs_and_FPCA_over_Years.png'
        plt.savefig(pathw)
        plt.close()

        return