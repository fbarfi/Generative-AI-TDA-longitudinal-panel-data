

import os
os.environ["DGLBACKEND"] = "pytorch"

from SubGraphX_python_codes.deployingsubgraphx import DeploySubGraphXClass

if __name__ == "__main__":

    first_year, last_year = 1946, 2010

    
    h_dim_list = ['H_1', 'H_2']

    h_dim = h_dim_list[0]
    nepochs = 200 #int(input("Enter number of epochs used for SubGraphX training (e.g., 200): "))

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    data_prepared = os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/")
    data_original = os.path.join(BASE_DIR, "project_data", "Data","Data_original/")
    Homology =  os.path.join(BASE_DIR, "project_data", "Data", "Homology/")
    folder_results =  os.path.join(BASE_DIR, "project_data", "Results", "SubGraphX_results/")

    args = {
        'first_year': first_year,
        'last_year': last_year,
        'h_dim_list': h_dim_list,
        'h_dim': h_dim,
        'data_original': data_original,
        'data_prepared': data_prepared,
        'Homology': Homology,
        'folder_results': folder_results,
        'n_years': [year for year in range(first_year, last_year+1)],
    }
    deploy = DeploySubGraphXClass(args)

    edges = deploy.ReConstructDatafromSubgraph()
