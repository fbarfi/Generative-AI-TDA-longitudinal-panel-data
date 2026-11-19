
import pandas as pd
import os

from gudhi_python_codes.gudhi_persistence_representations_class import GudhiPersistenceRepresentationsClass
from gudhi_python_codes.gudhi_persistence_diagrams_class import GudhiPersistenceDiagramsClass

if __name__ == "__main__":

    """
    Using Dyadic Data Augmented with Normalized IO Memberships
    Compute Persistence Diagrams, Persistence Images (saved as images)
    and Princicpal Functional Components for Persistence Images

    Parameters for persistence homology
    """
    homology_dimensions = [0,1,2]
    beginYear, endYear = 1946, 2010

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    args = {
        'beginYear': beginYear,
        'endYear':endYear,
        'folder': os.path.join(BASE_DIR, "project_data", "Data", "Data_original/"),
        'folder_prep': os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/"),
        'Data': os.path.join(BASE_DIR, "project_data", "Data/"),
        'Homology': os.path.join(BASE_DIR, "project_data", "Data", "Homology/"),
        'C': 5. ,
        'p': 3,
        'bins': 100, 
        'max_edge_length': 12.0,
        'max_dimension':  3,
        'homology_dimensions': homology_dimensions,
        'n_components':  3,
        'beginYear':  beginYear,
        'endYear':  endYear,
        'n_years': range(beginYear,endYear+1),
    }


    """
    Comoputing and Saving PDs
    """
    gudhi_PD_class = GudhiPersistenceDiagramsClass(args)
    gudhi_PD_class.grouping_PDs()
    print('Done computing persistence diagrams for all years!')

    gudhi_rep_class = GudhiPersistenceRepresentationsClass(args)
    gudhi_rep_class.saving_combined_results()

    print('Done computing persistence images for all years!')

    df_list = []
    for year in args['n_years']:
        filename = args['Homology'] + f'IO_data_hom/IO_data_hom_H_012_{year}.csv'
        df_list.append(pd.read_csv(filename).reset_index(drop=True))

    pd.concat([*df_list],axis = 0).reset_index(drop=True).to_csv(
        args['Homology'] + f'IO_data_hom_H_012_{args["beginYear"]}_{args["endYear"]}.csv',
        index=False)

    """
    Computing FPCA and saving
    """
    gudhi_rep_class.FPCA_()

    print('Done computing FPCA for all years!')



    """
    Computing entropy and saving
    """
    gudhi_rep_class.persistence_entropy()

    print('Done computing persistence  entropy for all years!')
