

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = StandardScaler()

from bayesian_python_codes.obtainresults import ObtainResults

def parameters_fn(model_type):

    n_iter = 2000
    tune = 1000

    scaling_MinMax = True
    if scaling_MinMax:
        scale = 'MinMax'
        scaler = MinMaxScaler()
    else:
        scale = 'Standard'
        scaler = StandardScaler()


    mcmc_config = {
        "draws": n_iter,
        "chains": 4,
        "tune": tune,
        "random_seed": 8927,
        "target_accept": 0.98,
    }

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    data_folder = os.path.join(BASE_DIR, "project_data", "Data/")
    data_prepared =  os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/")
    folder = os.path.join(BASE_DIR, "project_data","Results", "Bayesian_results", f"MCMC_{model_type}/")

    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")

    beginYear, endYear = 1946, 2010

    args = {
        'model_type': model_type,
        'beginYear': beginYear,
        'endYear': endYear,
        'SAMPLE_KWARGS': mcmc_config,
        'idata_kwargs': {"log_likelihood": True},
        'scaler': scaler,
        'folder': folder,
        'data_folder': data_folder,
        'data_prepared': data_prepared,
    }

    args['n_years'] = list(range(args['beginYear'], args['endYear'] + 1))
    print(f'\n-------- draws = {n_iter}, tuning = {tune}, MCMC chains = {mcmc_config["chains"]}')

    return args


if __name__ == "__main__":

    model_configs = {
        'bare': { 'var_names': ['β', 'ψ'],
            'vars': ['majpower', 'allies', 'contgkf', 'smldep', 'lcaprat', 'lgmidist',
                'smldmat','ψ' ]
        },
        'sociomatrix': {
            'var_names': ['β', 'ρ', 'ψ'],
            'vars': ['majpower', 'allies', 'contgkf', 'smldep', 'lcaprat', 'lgmidist',
                    'smldmat', 'ρ', 'ψ']
        },
        'sociomatrix_homology': {
            'var_names': ['β', 'ρ', 'ζ', 'δ', 'ψ'],
            'vars': ['majpower', 'allies', 'contgkf', 'smldep', 'lcaprat', 'lgmidist',
                    'smldmat', 'ρ', 'H_1', 'H_2', 'δ_H_1', 'δ_H_2','ψ']
        },
        'sociomatrix_homology_latent': {
            'var_names': ['β', 'ρ', 'ζ', 'δ', 'ξ', 'θ', 'ψ'],
            'vars': ['majpower', 'allies', 'contgkf', 'smldep', 'lcaprat', 'lgmidist',
                    'smldmat', 'ρ', 'H_1', 'H_2', 'δ_H_1', 'δ_H_2', 'L0', 'L1',
                    'θ_L0', 'θ_L1', 'ψ']
        }
    }

    for model_type, config in model_configs.items():
        print(f'\nProceeding with {model_type} model')

        args = parameters_fn(model_type)

        args['var_names'] = config['var_names']
        args['vars'] = config['vars']

        obtain_results = ObtainResults(args)

        plot_and_save_results = obtain_results.plot_save_func()
        print(f'Done for {model_type} model!')