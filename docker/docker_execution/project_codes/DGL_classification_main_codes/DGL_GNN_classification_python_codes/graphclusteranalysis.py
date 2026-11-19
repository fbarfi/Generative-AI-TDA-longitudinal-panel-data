import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from .graphsdatamodule import GraphsDataModule
from .rungnnmodel import RunGNNModel

class GraphClusterAnalysis():
    def __init__(self, args):
        self.args = args
        self.phases = ['train', 'test']
        self.out_channels = args['out_channels']
        self.data_original = args['data_original']
        self.folder = args['folder'] + 'plots/'
        self.models_folder = args['folder'] +'GNN_models/'
        self.countries = pd.read_csv(self.data_original+'COW_country_codes.csv',
                                     usecols = ['ccode', 'stateabb'])
        self.data_module = GraphsDataModule(args)
        self.data_module.setup()
    
    def get_data(self):        
        train_data = self.data_module.train_data
        val_data = self.data_module.val_data
        test_data = self.data_module.test_data
        return train_data, val_data, test_data

    def loading_model(self):
        args = self.args
        checkpoint_path = args['models_folder'] + 'saved_model.ckpt'
        model = RunGNNModel.load_from_checkpoint(
            checkpoint_path = checkpoint_path,
            model=RunGNNModel(args),
            args=args
        )
        return model
    
    
    def convert_ccode_to_name(self, ccode):
        name = self.countries.loc[self.countries['ccode'] == ccode, :]['stateabb'].values[0]
        return name
    

    def assign_clusters_to_graphs(self, dataloader):
        model = self.loading_model()
       
        graph_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                g, labels = batch
                outputs, _ = model(batch)
                graph_embeddings.append(outputs.detach().cpu().numpy())

        graph_embeddings = np.concatenate(graph_embeddings)
        
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(graph_embeddings)
        
        distances = pairwise_distances(graph_embeddings, kmeans.cluster_centers_)

        graph_probabilities = np.exp(-distances) 
        graph_probabilities = graph_probabilities / graph_probabilities.sum(axis=1, keepdims=True) 

        return graph_probabilities

    def extract_years(self, data):
        years = []
        for g,label in data:
            years.append(g.year.item())  
        return np.array(years)
    
    def analyze_probabilities(self):
        train_data, val_data, test_data = self.get_data()
        dataloader = {'all': train_data + val_data + test_data}

        graph_probabilities = self.assign_clusters_to_graphs(dataloader['all'])
        graph_years = self.extract_years(dataloader['all'])

        return graph_probabilities, graph_years

    
    def plot_probabilities(self):
        n_clusters = 4
        nyears = self.args['nyears']
        probabilities, all_years = self.analyze_probabilities()
    
        all_rows = []
        for i in range(len(probabilities)):
            all_rows.append({
                'Year': all_years[i],
                'Cluster_Probabilities': probabilities[i] 
            })

        df = pd.DataFrame(all_rows)   
           
        unique_years = np.sort(np.unique(df['Year']))
        year_probabilities = np.zeros((len(unique_years), n_clusters))

        for i, year in enumerate(unique_years):
            year_data = df[df['Year'] == year]
            year_probs = np.vstack(year_data['Cluster_Probabilities'].values)
    
            if year_probs.shape[1] < n_clusters:
                padded_probs = np.zeros((year_probs.shape[0], n_clusters))
                padded_probs[:, :year_probs.shape[1]] = year_probs
                year_probs = padded_probs
            elif year_probs.shape[1] > n_clusters:
                year_probs = year_probs[:, :n_clusters]

            year_probabilities[i] = np.mean(year_probs, axis=0)
        
        fig,ax = plt.subplots(figsize=(12, 8))
    
        plt.stackplot(unique_years, year_probabilities.T, 
                labels=[f'Cluster {i}' for i in range(n_clusters)],
                alpha=0.7)

        plt.xlabel('Year',fontsize = 14)
        plt.ylabel('Probability',fontsize = 14)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=n_clusters, 
                   bbox_to_anchor=(0.5, -0.025), fontsize=14)

        plt.suptitle('Cluster Membership Probabilities', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  

        plt.savefig(self.folder + 'train_test_class_probabilities_proportions_over_years.png', 
                    dpi = 300, bbox_inches='tight')
        plt.close()
         
        return 

    