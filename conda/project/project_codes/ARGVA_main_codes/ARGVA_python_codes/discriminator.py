import torch
import pytorch_lightning as pl
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class Discriminator(pl.LightningModule):
    
    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        self.args = args
        self.in_channels = args['in_channels']
        self.hidden_channels = args['hidden_channels']
        self.num_edges_features = args['num_edges_features']
        self.out_channels = args['out_channels']
        self.num_layers = args['num_layers']
        
        self.conv1 = GATv2Conv(in_channels = self.in_channels,
                                out_channels = self.hidden_channels,
                              edge_dim = self.num_edges_features,
                              )                          
        self.conv2 = GATv2Conv(in_channels = self.hidden_channels,
                                out_channels = self.hidden_channels,
                              edge_dim = self.num_edges_features,
                              )                
        self.conv3 =GATv2Conv(in_channels = self.hidden_channels,
                                out_channels = self.hidden_channels,
                              edge_dim = self.num_edges_features,
                             )
        self.linear = nn.Linear(self.hidden_channels, self.out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        edge_index = edge_index.clone().detach().long()
        
        x = self.conv1(x , edge_index, edge_attr)
                                 
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
       
        for _ in range(self.num_layers):
            x= self.conv2(x , edge_index, edge_attr)
                                       
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            
        x = self.conv3(x, edge_index, edge_attr)
        x = self.linear(x)
        
        return x 