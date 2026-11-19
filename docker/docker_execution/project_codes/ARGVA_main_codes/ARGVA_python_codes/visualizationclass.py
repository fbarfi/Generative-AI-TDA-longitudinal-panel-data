import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torch_geometric.utils import subgraph
from scipy.stats import kde
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.metrics.pairwise import pairwise_distances_argmin

import torch
from .argvamodel import ARGVAModel
from .createdatamodule import CreateDataModule


class VisualizationClass():
    
    def __init__(self, args):
        
        super(VisualizationClass,self).__init__()
        self.args= args
        self.folder = args['folder']
        self.n_epochs = args['n_epochs']
        
    def loading_model(self, year):
        checkpoint_path = self.args['models_folder']+ f'/model_{year}_path.ckpt'
        model = ARGVAModel.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                                      year=year,args=self.args)
        return model
        
    def get_data(self,year):                                      
        datamodule = CreateDataModule(self.args,year)
        datamodule.setup()
        dataloaders = {
            "train":datamodule.train_dataloader(),
            "val":datamodule.val_dataloader(),
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
        
    def plot_points(self):
        labels = ['0','1']
        colors = [ 'red','blue']
        
        for year in self.args['specific_years']:
            model =self.loading_model(year)
            model.eval()
            
            dataloaders = self.get_data(year)
            fig, ax= plt.subplots(2,2, figsize=(12,12 ))
            ax = ax.flatten()
            for key,kk in zip(['train','test'],[0,2]):
                dataloader = self.get_data(year)[key]
                za_list, ya_list = [],[]
                for data in dataloader:
                    mask = getattr(data, f"{key}_mask").bool()
                    edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
                
                    z = self.z_parameterize(model.encoder, 
                                            data.x[mask],
                                            edge_index_masked,
                                            edge_attr_masked)
                    
                    za_list.append(z.detach().numpy())
                    
                    ya_list.append(data.y[mask].detach().numpy())

                z_array = np.vstack(za_list)
                y = np.hstack(ya_list)

                """ PCA latentspace plot """
                z_pca = PCA(n_components=2).fit_transform(z_array)
                for i in range(2):
                    ax[kk].scatter(z_pca[y == i, 0], z_pca[y == i, 1],s=20, 
                                  color=colors[i],
                                  label = f'disp = {i}')
                x = z_pca[:, 0]
                y_contour = z_pca[:, 1]
                nbins = 100
                k = kde.gaussian_kde([x, y_contour])
                xi, yi = np.mgrid[x.min():x.max():nbins*1j, y_contour.min():y_contour.max():nbins*1j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
                contour = ax[kk].contour(xi, yi, zi.reshape(xi.shape), levels=10, cmap="plasma")
                ax[kk].clabel(contour, inline=True, fontsize=8)
            
                ax[kk].set_title(f'{key}: PCA Latentspace features', fontsize=14)
                ax[kk].set_xlabel('z_0')
                ax[kk].set_ylabel('z_1')

                """ Kmeans latenspace plot """
                n_clusters = 5
                k_means = KMeans(init="k-means++", 
                             n_clusters=n_clusters, 
                             max_iter=2000, 
                             random_state=0, 
                             n_init='auto')
                k_means.fit(z_pca)
            
                k_means_cluster_centers = k_means.cluster_centers_
                k_means_labels = pairwise_distances_argmin(z_pca, k_means_cluster_centers)
              
                h = 0.001
                x_min, x_max = z_pca[:, 0].min() - 1.5, z_pca[:, 0].max() + 1.5
                y_min, y_max = z_pca[:, 1].min() - 1.5, z_pca[:, 1].max() + 1.5
                
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                xx = xx.astype('float32')
                yy = yy.astype('float32')
                Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
        
                colormap = plt.get_cmap('tab10')
                ax[kk+1].imshow(Z, interpolation="nearest", extent=(xx.min(), 
                                xx.max(), yy.min(), yy.max()), cmap=colormap, aspect="auto", 
                                origin="lower")
                ax[kk+1].plot(z_pca[:, 0], z_pca[:, 1], "k.", label='Data Points', alpha=0.5)
        
                centroids = k_means.cluster_centers_
                ax[kk+1].scatter(centroids[:, 0], centroids[:, 1], marker="*", s=100, 
                            linewidths=3, color="w", zorder=30, label='Centroids')
                
                legend_patches = [mpatches.Patch(color='black', label='Data Points')]
                legend_patches.append(mpatches.Patch(color='white', label='Centroids'))
                ax[kk+1].legend(handles=legend_patches, loc='center')
                
                x_min, x_max = centroids[:, 0].min() - 0.1, centroids[:, 0].max() + 0.1
                y_min, y_max = centroids[:, 1].min() - 0.1, centroids[:, 1].max() + 0.1
        
                ax[kk+1].set_xlim(x_min, x_max)
                ax[kk+1].set_ylim(y_min, y_max)
                ax[kk+1].set_xticks(())
                ax[kk+1].set_yticks(())
    
                for i, centroid in enumerate(centroids):
                    ax[kk+1].annotate(f'Centroid {i}', xy=(centroid[0], centroid[1]), xytext=(4, 4), 
                                 textcoords="offset points", ha='right', va='bottom', color="w", 
                                 fontsize=12)
                ax[kk+1].set_title(f'{key}: latentspace kmeans clusters', fontsize = 14)
            
            plt.suptitle(f'Graph Embedding, Year: {year}',fontsize =18)
            plt.tight_layout();
    
            path_name = self.folder + f'latentspace/latentSpace_embedding_{year}.png'
            plt.savefig(path_name)
            
            plt.close();
        
        return