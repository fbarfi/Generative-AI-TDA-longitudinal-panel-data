import torch
import pytorch_lightning as pl
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F

class Encoder(pl.LightningModule):
    def __init__(self, args):
        super(Encoder,self).__init__()
        
        self.args = args
        self.hidden_channels = args['hidden_channels']       
        self.in_channels = args['in_channels']
        self.out_channels = args['out_channels']
        self.num_edges_features = args['num_edges_features']
        self.num_layers = args['num_layers']
        self.conv1 = GATv2Conv(in_channels = self.in_channels,
                                out_channels = self.hidden_channels,
                                edge_dim = self.num_edges_features,
                                share_weights=True)
                              
        self.conv2 = GATv2Conv(in_channels = self.hidden_channels,
                                out_channels = self.hidden_channels,
                              edge_dim = self.num_edges_features,
                              share_weights=False)
                             
        self.conv3 =GATv2Conv(in_channels = self.hidden_channels,
                                out_channels = self.hidden_channels,
                              edge_dim = self.num_edges_features,
                              share_weights=True)
                     
        self.linear = nn.Linear(self.hidden_channels, self.out_channels)
                    
        self.conv_mu = GATv2Conv(in_channels = self.out_channels, 
                                 out_channels = self.out_channels,
                                edge_dim = self.num_edges_features,
                                 share_weights=True)
        self.conv_logstd = GATv2Conv(in_channels = self.out_channels, 
                                    out_channels = self.out_channels,
                                      edge_dim = self.num_edges_features,
                                     share_weights=True)
        
    def forward(self, x, edge_index, edge_attr):
        att_weights_all_layers = []
        edge_index = edge_index.clone().detach().long()
        
        x, att_weights = self.conv1(x, edge_index, edge_attr, 
                                    return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        att_weights_all_layers.append(att_weights[1])

        for _ in range(self.num_layers):
            x, att_weights = self.conv2(x, edge_index, edge_attr, 
                                        return_attention_weights=True)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            att_weights_all_layers.append(att_weights[1])
        
        x, att_weights = self.conv3(x, edge_index, edge_attr, 
                                    return_attention_weights=True)
        att_weights_all_layers.append(att_weights[1])
        
        x = self.linear(x)
        
        mu, att_weights = self.conv_mu(x, edge_index, edge_attr, 
                                       return_attention_weights=True)
        att_weights_all_layers.append(att_weights[1])
        
        logstd, att_weights = self.conv_logstd(x, edge_index, edge_attr, 
                                               return_attention_weights=True)
        att_weights_all_layers.append(att_weights[1])
        
        mean_att_weights = torch.mean(torch.stack(att_weights_all_layers,dim=1), dim=1)
        
        return mu, logstd, (att_weights[0], mean_att_weights)