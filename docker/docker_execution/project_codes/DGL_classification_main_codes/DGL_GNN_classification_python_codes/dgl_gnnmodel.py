import dgl
from dgl.nn import AvgPooling, EdgeGATConv
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(pl.LightningModule):
    def __init__(self, args):
        super(GNNModel, self).__init__()

        self.args = args
        self.hidden_channels = args['hidden_channels']       
        self.in_feats = args['in_channels']
        self.out_feats = args['out_channels']
        self.edge_feats = args['num_edge_features']
        self.num_heads = args['heads']
        self.num_hidden_layers = args['num_hidden_layers']
        self.dropout = args['dropout']
        self.save_hyperparameters(args)
        self.activation = nn.Tanh()  
        self.feat_drop = args['feat_drop']

        self.conv1 = EdgeGATConv(
            in_feats=self.in_feats,
            out_feats=self.hidden_channels, 
            num_heads=self.num_heads,
            edge_feats=self.edge_feats,
            allow_zero_in_degree=True,
            feat_drop = self.feat_drop,
            bias=True,
            activation=self.activation
        )
        
        self.layers = nn.ModuleList([
            EdgeGATConv(
                in_feats=self.hidden_channels * self.num_heads,
                out_feats=self.hidden_channels,  
                num_heads=self.num_heads,
                edge_feats=self.edge_feats,
                allow_zero_in_degree=True,
                feat_drop = self.feat_drop,
                bias=True,
                activation=self.activation
            ) for _ in range(self.num_hidden_layers)
        ])
        
        self.conv3 = EdgeGATConv(
            in_feats=self.hidden_channels * self.num_heads,
            edge_feats=self.edge_feats, 
            out_feats=self.hidden_channels,   
            num_heads=self.num_heads,
            allow_zero_in_degree=True,
            feat_drop = self.feat_drop,
            bias=True,
            activation=self.activation
        )

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.pool = AvgPooling()
        self.linear = nn.Linear(self.hidden_channels * self.num_heads, self.out_feats)

    def forward(self, g, x, efeat):
        all_att_weights = []
    
        x, att_weights = self.conv1(g, x, efeat, get_attention=True)
        all_att_weights.append(att_weights)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = F.relu(x)
        x = self.dropout1(x)
        
        for i, conv in enumerate(self.layers):
            x, att_weights = conv(g, x, efeat, get_attention=True)
            all_att_weights.append(att_weights)

            x = torch.flatten(x, start_dim=1, end_dim=2)
            x = F.relu(x)
            x = self.dropout1(x)

        x, att_weights = self.conv3(g, x, efeat, get_attention=True)
        all_att_weights.append(att_weights)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = F.relu(x)
        x = self.dropout1(x)
      
        x = F.relu(self.pool(g, x)) 
        x = self.linear(x)
        
        stacked_att_weights = torch.stack(all_att_weights, dim=1)
        mean_att_weights = torch.mean(stacked_att_weights, dim=(1, 2))  
        
        mean_att_weights = mean_att_weights.squeeze()
        mean_att_weights_expanded = mean_att_weights.unsqueeze(0).expand(2, -1)
        
        return x, mean_att_weights_expanded

