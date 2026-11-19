import torch
import numpy as np
import os
import sys

from DGL_GNN_classification_python_codes.graphclusteranalysis import GraphClusterAnalysis

if __name__ == "__main__":

    first_year , last_year = 1946, 2010

    nepochs = 1000

    learning_rate = 1e-4
    heads = 1
    batch_size = 5
    hidden_channels = 128
    num_hidden_layers = 100
    monitor = "val_loss"
    mode = "min"
    num_classes = 4
    weight_decay = 1.e-4
    patience = 20
    dropout = 0.1
    feat_drop = 0.1
    val_ratio = 0.2
    test_ratio = 0.1
    gradient_clip_val = 1.

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))   

    folder = os.path.join(BASE_DIR, "project_data", "Results", "DGL_GNN_Classification/")
    Data = os.path.join(BASE_DIR, "project_data", "Data/")
    data_prepared = os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/")
    data_original = os.path.join(BASE_DIR, "project_data", "Data","Data_original/")
    data_homology =  os.path.join(BASE_DIR, "project_data", "Data", "Homology/")

    args = {
        'seed': 42,
        'first_year': first_year,
        'last_year': last_year,
        'features': ['disp_weight', 'power'],
        'edge_features': ['allies', 'contgkf', 'smldep',
                          'lcaprat', 'lgmidist', 'smldmat', 'weight'],
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'nepochs': nepochs,
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'patience': patience,
        'feat_drop': feat_drop,
        'out_channels': 4,
        'heads': heads,
        'num_hidden_layers': num_hidden_layers,
        'dropout': dropout,
        'gradient_clip_val': gradient_clip_val,
        'learning_rate': learning_rate,
        'h_dim': 'H_1',
        'device': torch.device('cpu'),
        'accelerator': 'cpu',
        'monitor':monitor,
        'mode':mode,
        'weight_decay': weight_decay,
        'num_classes': num_classes,
        'folder': folder,
        'Data': Data,
        'data_original': data_original,
        'data_prepared': data_prepared,
        'data_homology': data_homology,
    }

    args['models_folder'] = args['folder'] + 'GNN_models/'
    args['num_edge_features'] = len(args["edge_features"])
    args['in_channels'] = len(args['features'])
    args['nyears'] = range(args['first_year'], args['last_year'] + 1)

    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    print("[INFO] Running cluster analysis...")
    
    analysis = GraphClusterAnalysis(args) 
    analysis.plot_probabilities()

    print("✅ Cluster analysis complete.")

    

    


    
