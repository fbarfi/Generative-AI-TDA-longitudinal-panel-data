
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler1 = MinMaxScaler()
scaler2 = StandardScaler()

from bayesian_subgraphx_python_codes.bayesian_model import BayesianModel
from bayesian_subgraphx_python_codes.save_results import SaveResults


if __name__ == "__main__":

    """
    Choice in scaling float covariates
    """
    scaling_MinMax = True

    if scaling_MinMax:
        scale = 'MinMax'
        scaler = MinMaxScaler()
    else:
        scale = 'Standard'
        scaler = StandardScaler()

    """
    Parameters for MCMC
    """
    n_iter = 15_000
    tune = 5_000

    chains = 4
    target_accept = 0.98
    random_seed = 8927
    SAMPLE_KWARGS = {
        "draws": n_iter,
        "chains": chains,
        "tune": tune,
        "random_seed": random_seed,
        "target_accept": target_accept,
        "progressbar": True,
    }

    idata_kwargs = {"log_likelihood": True}

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    subgraphx_folder = os.path.join(BASE_DIR, "project_data","Results", "SubgraphX_results/")
    folder_prepared = os.path.join(BASE_DIR, "project_data","Data", "Data_prepared/")
    folder = os.path.join(BASE_DIR, "project_data","Results", "Bayesian_results", "MCMC_SubGraphX/")

    os.makedirs(folder, exist_ok=True)

    beginYear, endYear = 1946, 2010

    """
    nepochs should be equal to nepochs used to train the SubGraphX deep learning classification, so as
    to retireve the corresponding subgraph data for each year
    """
    nepochs = 200

    args = {
        'beginYear': beginYear,
        'endYear': endYear,
        'nepochs': nepochs,
        'SAMPLE_KWARGS': SAMPLE_KWARGS,
        'idata_kwargs': idata_kwargs,
        'scaler': scaler,
        'folder': folder,
        'var_names': ['β', 'ψ'],
        'vars': ['majpower', 'allies', 'contgkf', 'smldep', 'lcaprat', 'lgmidist',
                 'smldmat', 'ψ'],
         }
    args['n_years'] = range(args['beginYear'], args['endYear'] + 1)
    
    args['subgraphx_folder'] = subgraphx_folder
    args['folder_prepared'] = folder_prepared
    args['folder'] = folder

    """
    Initialize and Train Bayesian Model and Perform out-of-sample testing
    """
    bayesian_model = BayesianModel(args)
    bayesian_model.run_model()

    """
    Save and Plot Results
    """
    save_results = SaveResults(args)
    save_results.plot_save_func()
    save_results.plot_test_roc_auc()
