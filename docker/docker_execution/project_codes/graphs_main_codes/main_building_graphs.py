

import pandas as pd
import os

from graphs_python_codes.creategraphsdata import CreateGraphsData

def create_homology_subdirs(base_dir, h_list):
        homology_dir = os.path.join(base_dir, "project_data", "Data", "Homology", "Homology_categorized")
        os.makedirs(homology_dir, exist_ok=True)

        for h in h_list:
            subdir = os.path.join(homology_dir, h)
            os.makedirs(subdir, exist_ok=True)


if __name__ == "__main__":

    nodes_features = ['disp_weight','power']
    edges_features = ['allies', 'contgkf','smldep', 'lcaprat',
                      'lgmidist', 'smldmat', 'weight']

    beginYear , endYear = 1946, 2010

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    n_components = 3
    args = {
        'beginYear': beginYear,
        'endYear': endYear,
        'num_cat': 5,
        'folder': os.path.join(BASE_DIR, "project_data", "Data", "Data_original/"),
        'folder_prep': os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/"),
        'Data': os.path.join(BASE_DIR, "project_data", "Data/"),
        'Homology': os.path.join(BASE_DIR, "project_data", "Data", "Homology/"),
        'H_list':  [f'H_{i}' for i in range(1,n_components+1)],
        'nodes_features': nodes_features,
        'edges_features': edges_features,
    }
   
    create_homology_subdirs(BASE_DIR, args['H_list'])

    file_path = args['folder_prep'] +  'all_vars_1885_2010.csv'
    data_file =pd.read_csv(file_path)
    data_file = data_file.loc[(data_file['year'] > beginYear -1) 
                              & (data_file['year'] < endYear +1), :].reset_index(drop = True)
    args['file_vars'] = data_file

    args['n_years'] = [year for year in range(args['beginYear'], args['endYear']+1)]


    graphs_class = CreateGraphsData(args)
    graphs = graphs_class.save_graphs_data()
