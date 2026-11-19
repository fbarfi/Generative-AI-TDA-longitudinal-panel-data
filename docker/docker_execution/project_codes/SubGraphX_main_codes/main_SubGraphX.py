


import os
import warnings
import torch

os.environ["DGLBACKEND"] = "pytorch"
device = torch.device("cpu")

import warnings
warnings.filterwarnings("ignore", message=".*MPS available but not used.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*target size.*")
warnings.filterwarnings("ignore", message=".*Experiment logs directory.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists and is not empty.*")
warnings.filterwarnings("ignore", message=".*TPU available:.*")
warnings.filterwarnings("ignore", message=".*IPU available:.*")
warnings.filterwarnings("ignore", message=".*HPU available:.*")

from SubGraphX_python_codes.dglgraphsdataset import DGLGraphsDatasetClass
from SubGraphX_python_codes.runningsubgraphexplainer import RunningSubgraphExplainer
from SubGraphX_python_codes.plottingsubgraphx import PlottingSubgraphXClass
from SubGraphX_python_codes.modeltrainer import ModelTrainer
from SubGraphX_python_codes.gnnutilities import GNNUtilities



if __name__ == "__main__":

    first_year, last_year = 1985, 2010
    n_years = list(range(first_year,last_year +1))

    """ Homology """
    h_dim_list = ['H_1', 'H_2']
    device = torch.device('cpu')
    specific_years = [1955,1975,1995,2010]

    nepochs = 200
    num_rolouts = 10

    m = int(nepochs / 10)
    patience = 10

    out_feats = 4
    hidden_dim = 64
    batch_size = 5
    num_layers = 10

    num_child = 12 
    node_min = 15
    num_hops = 5
    num_heads = 6
    shapley_steps = 80
    coef = 10.

    learning_rate = 1.e-5
    num_classes = 4
    dropout_rate = 0.3

    val_ratio = 0.20
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    data_prepared = os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/")
    data_original = os.path.join(BASE_DIR, "project_data", "Data","Data_original/")
    Homology =  os.path.join(BASE_DIR, "project_data", "Data", "Homology/")

    args = {
        'n_years': n_years,
        'nepochs': nepochs,
        'first_year': first_year,
        'last_year': last_year,
        'specific_years': specific_years,
        'h_dim_list': h_dim_list,
        'h_dim': h_dim_list[0],
        'out_feats': out_feats,
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'num_layers': num_layers,
        'log': True,
        'num_rollouts': num_rolouts,
        'num_child': num_child,
        'node_min': node_min,
        'num_hops': num_hops,
        'num_heads': num_heads,
        'shapley_steps': shapley_steps,
        'coef': coef,
        'device': device,
        'accelerator': 'cpu',
        'mode': 'min',
        'monitor':'val_loss',
        'patience': patience,
        'learning_rate': learning_rate,
        'num_classes': num_classes,
        'dropout_rate': dropout_rate,
        'residual':True,
        'Homology': Homology, 
        'data_prepared': data_prepared, 
        'data_original': data_original,
    }

    dataset = DGLGraphsDatasetClass(args)
    graphs = dataset.graphs

    num_graphs = len(graphs)
    print('Number of graphs:', num_graphs)

    g = graphs[0]
    in_feats = g.ndata['feat'].detach().numpy().shape[1]
    edge_feats = g.edata['efeat'].detach().numpy().shape[1]

    args.update({
        'num_graphs': num_graphs,
        'val_ratio': val_ratio,
        'in_feats': in_feats,
        'edge_feats': edge_feats,
    })

    args['folder_results'] = os.path.join(BASE_DIR, "project_data", "Results", "SubGraphX_results/")


    """ Creating folders for results """
    util = GNNUtilities(args)
    util.create_results_folders()

    """
    training and saving model
    """
    trainer = ModelTrainer(args)
    trainer.train_model()

    """ Running explainer and saving resuting subgraphs """
    run_explainer = RunningSubgraphExplainer(args)

    message_two = run_explainer.using_SubGraphXExplainer()
    print(message_two)

    plotting_graphs = PlottingSubgraphXClass(args)
    message_three = plotting_graphs.plot_subgraph()

    print(message_three)
