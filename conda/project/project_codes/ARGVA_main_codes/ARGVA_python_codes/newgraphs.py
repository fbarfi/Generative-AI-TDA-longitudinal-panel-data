from torch_geometric.utils import subgraph
from torch_geometric.utils import degree, to_networkx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import torch

from .argvamodel import ARGVAModel
from .createdatamodule import CreateDataModule

class NewGraphs():
     
    def __init__(self, args,year): 
        super(NewGraphs, self).__init__()  
        self.args = args
        self.latent_size = args['latent_size']
        self.folder = args['folder']
        self.year = year
        
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
    
    def graph_properties(self, G=None, data=None):
        if G is None and data is not None:
            G = to_networkx(data, to_undirected=True)
        
        """
        Node and Edge Distributions
        """
        node_degrees = [val for (node, val) in G.degree()]
        
        """
        Clustering Coefficients
        """
        clustering_coefficients = nx.clustering(G)
        
        """
        Average Path Length
        """
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            """ Get the largest connected component """
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
        
        """
        Display some of the computed metrics
        """
        Average_Node_Degree = sum(node_degrees) / len(node_degrees)
        Average_Clustering_Coefficient = sum(
            clustering_coefficients.values()) / len(clustering_coefficients)
        Average_Path_Length = avg_path_length 
    
        return Average_Node_Degree, Average_Clustering_Coefficient, Average_Path_Length
         
    def z_parameterize(self, model, x, edge_index, edge_attr):
        mu, logvar,_ = model(x, edge_index, edge_attr)
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        z = mu + std * eps  
        return z
           
    def plotting_new_graphs(self, year, num_samples, threshold):
        """
        Generate A New Adjacency Matrix and its Corresponding Graph 
        with num_samples Nodes
        """
        torch.manual_seed(21)
        model = self.loading_model(self.year)
        model.eval()
        
        dataloader = self.get_data(year)['test']
        za_list= []
        for data in dataloader:
            mask = getattr(data, f"test_mask").bool()
            edge_index_masked, edge_attr_masked = self.masking_data(data, mask)
            z = self.z_parameterize(model.encoder, 
                                    data.x[mask],
                                    edge_index_masked,
                                    edge_attr_masked)
            za_list.append(z)  
        z = torch.concat(za_list,dim=0)
    
        graphs = []
        fig, ax = plt.subplots(figsize=(7,7))
     
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                            top=0.9, wspace=0.4, hspace=0.4)
        
        dx = torch.randn((num_samples, self.latent_size))
        
        adj = torch.matmul(z, z.t())
        adj_matrix = torch.sigmoid(adj).detach().numpy()  # > threshold).astype(int)
        
        G = nx.from_numpy_array(adj_matrix)
        AND, ACC, APL = self.graph_properties(G)
        
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)
        pos = nx.spring_layout(G)
        
        nx_G = nx.draw(G, pos, with_labels=True, node_color='lightblue',
                       edge_color='gray', ax=ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        padding = 0.1
        frame_width = xlim[1] - xlim[0] + padding * 2
        frame_height = ylim[1] - ylim[0] + padding * 2
        frame = patches.Rectangle((xlim[0] - padding, ylim[0] - padding), frame_width, 
                                  frame_height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(frame)
        path_n = f'ARGVA: New Graph \n AND = {AND:.3f}, ACC = {ACC:.3f}, '
        path_n += f'APL = {APL:.3f}, year {self.year}'
        ax.set_title(path_n)
       
        graphs.append(G)
        
        plt.tight_layout()
        
        path_name = f'generated_graphs/test_data_ARGVA_generated_graphs_{year}.png'
        plt.savefig(self.folder + path_name)
        plt.close()
    
        return graphs