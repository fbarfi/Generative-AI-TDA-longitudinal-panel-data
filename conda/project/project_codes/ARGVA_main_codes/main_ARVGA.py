

import os
import torch
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.WARNING)
logging.disable(logging.CRITICAL)

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

from ARGVA_python_codes.trainingtestingclass import TrainingTestingClass
from ARGVA_python_codes.generalutilities import GeneralUtilities
from ARGVA_python_codes.augmentingpaneldata import AugmentingPanelData
from ARGVA_python_codes.featuresextraction import FeaturesExtraction
from ARGVA_python_codes.newgraphs import NewGraphs
from ARGVA_python_codes.visualizationclassii import VisualizationClassII
from ARGVA_python_codes.visualizationclass import VisualizationClass

if __name__ == "__main__":

    h_dim = 'H_1'
    node_features = ['disp_weight','power']
    edges_features = ['allies', 'contgkf','smldep', 'lcaprat',
                      'lgmidist', 'smldmat', 'weight']


    specific_years = [1980, 1990, 2000, 2010]
    specific_states = ['USA', 'UKG', 'RUS', 'IRQ']

    n_epochs = 500

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    folder = os.path.join(BASE_DIR, "project_data", "Results", "ARGVA_results/")
    Data = os.path.join(BASE_DIR, "project_data", "Data/")
    data_prepared = os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/")
    data_original = os.path.join(BASE_DIR, "project_data", "Data","Data_original/")
    data_homology =  os.path.join(BASE_DIR, "project_data", "Data", "Homology/")

    models_folder = folder + 'saved_ARGVA_models/'
    
    result_subfolders = [
            'classifications/',
            'latentspace/',
            'generated_graphs/',
            'losses/',
            'losses/plots/',
            'saved_ARGVA_models/',
            'latentspace/pca_kde/',
            'latentspace/latentspace_csv/',
        ]

    first_year, last_year = 1946, 2010
    nyears =  range(first_year,last_year+1)

    channels_size = 128
    num_classes = 4

    args = {
        'batch_size': 10,
        'num_neighbors': [5]*3,
        'specific_years': specific_years,
        'specific_states': specific_states,
        'first_year': first_year,
        'last_year': last_year,
        'h_dim': h_dim,
        'node_features': node_features,
        'num_features': len(node_features),
        'edges_features' : edges_features,
        'num_edges_features': len(edges_features),
        'hidden_size':channels_size,
        'latent_size': channels_size,
        'n_epochs': n_epochs,
        'learning_rate': 1.e-4,
        'in_channels': len(node_features),
        'out_channels': len(node_features),
        'hidden_channels': channels_size,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'perplexity': 20,
        'device': torch.device('cpu'),
        'folder': folder,
        'models_folder': models_folder,
        'data_original':data_original,
        'data_prepared': data_prepared,
        'data_homology': data_homology,
        'subfolders': result_subfolders,
        'accelerator': 'cpu',
        'nyears': nyears,
        'num_classes': num_classes,
        'num_layers': 20,
        'tda_dim': len(nyears),
    }

    """
    Create folders and subfolders for storing results
    """
    util = GeneralUtilities(args)
    util. create_results_folders()

    """ Training and saving model """

    for year in tqdm(nyears):
        print(f"Training and Saving Model for year {year}")
        train_test_class = TrainingTestingClass(args,year)

        train_test_class.training_testing_model()

    print("Done Training and Saving Model")

    """
    Visualization of preliminary results
    """
    VizClass = VisualizationClass(args)
    VizClass.plot_points()

    """
    Generating New Graphs
    """
    for year in tqdm(args['specific_years']):
        new_graphs = NewGraphs(args,year)

        num_samples = 20
        threshold = 0.5
        graphs = new_graphs.plotting_new_graphs(year, num_samples, threshold)

    """
    Extracting and analyzing latent features
    """
    feature_extract = FeaturesExtraction(args)
    feature_extract.building_nodes_latent_features()
    feature_extract.pca_analysis()
    feature_extract.kmeans_cluster_latent_space()
    feature_extract.random_forest_classification()
    feature_extract.plot_latent_space()

    """ Augmenting data with latent variables """
    APD = AugmentingPanelData(args)
    APD.augmenting_data()

    """ Visulaizing different aspects of results """
    viz_classII = VisualizationClassII(args)
    viz_classII.average_proportion_nodes_in_kmeans_clusters()
    viz_classII.silhouette_scores()
    viz_classII.node_density_scores()
    viz_classII.multiple_cluster_analyses()
    viz_classII.cluster_cohesion_proportions_over_time()
    viz_classII.individual_cluster_cohesion_over_time()
    viz_classII.grouped_individual_cluster_cohesion_over_time()
    viz_classII.record_nodes_in_clusters()
    viz_classII.plot_node_probabilities()
