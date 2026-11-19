import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, EdgeGATConv


class EdgeGATConv_Model(nn.Module):
    def __init__(self, args):
        super(EdgeGATConv_Model, self).__init__()

        self.args = args
        self.in_feats = args['in_feats']
        self.edge_feats = args['edge_feats']
        self.out_feats = args['out_feats']
        self.num_heads = args['num_heads']
        self.hidden_dim = args['hidden_dim']
        self.num_layers = args['num_layers']
        self.activation = nn.Tanh()

        # First layer
        self.conv1 = EdgeGATConv(
            in_feats=self.in_feats, 
            edge_feats=self.edge_feats, 
            out_feats=self.hidden_dim,
            num_heads=self.num_heads,
            allow_zero_in_degree=True,
            bias=True,
            activation=self.activation
        )
        
      # Intermediate layers (conv2)
        self.layers = nn.ModuleList([
            EdgeGATConv(
                in_feats=self.num_heads * self.hidden_dim, 
                edge_feats=self.edge_feats, 
                out_feats=self.hidden_dim,
                num_heads=self.num_heads,
                allow_zero_in_degree=True,
                bias=True,
                activation=self.activation
            ) for _ in range(self.num_layers - 2)
        ])
        
        # Final layer (conv3)
        self.conv3 = EdgeGATConv(
            in_feats=self.num_heads * self.hidden_dim, 
            edge_feats=self.edge_feats, 
            out_feats=self.out_feats,
            num_heads=self.num_heads,
            allow_zero_in_degree=True,
            bias=True,
            activation=self.activation
        )
        
        self.pool = AvgPooling()
     
    def forward(self, graph, feat, efeat):
        # Apply the first convolutional layer
        feat = self.conv1(graph, feat, efeat)
        feat = torch.flatten(feat, start_dim=1, end_dim=2)
        feat = F.relu(feat)

        # Apply the intermediate convolutional layers
        for conv in self.layers:
            feat = conv(graph, feat, efeat)
            feat = torch.flatten(feat, start_dim=1, end_dim=2)
            feat = F.relu(feat)
        
        # Apply the final convolutional layer
        feat = self.conv3(graph, feat, efeat)
        feat = torch.flatten(feat, start_dim=1, end_dim=2)
        feat = F.relu(feat)
        
        # Pooling
        return F.relu(self.pool(graph, feat))

