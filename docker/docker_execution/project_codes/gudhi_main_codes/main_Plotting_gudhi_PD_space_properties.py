
import numpy as np
import os 

from gudhi_python_codes.gudhi_plotting_space_properties_class import GudhiPlottingPropertiesClass

if __name__ == "__main__":

    homology_dimensions = [0,1,2]
    beginYear,endYear =  1946, 2010
    n_years = range(beginYear,endYear+1)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    args = {
            'folder_prep': os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/"),
            'Homology': os.path.join(BASE_DIR, "project_data", "Data", "Homology/"),
            'endYear': endYear,
            'beginYear':beginYear,
            'homology_dimensions': homology_dimensions,
            'n_years': n_years,
            'n_components': 3,
           }

    plot_class = GudhiPlottingPropertiesClass(args)
    plot_class.persistence_entropy()
    plot_class.silhouette_scores()
    plot_class.silhouette_score_per_cluster_over_years()
    plot_class.average_silhouette_score_and_total_mids_over_years()
    plot_class.plot_mids_vs_silhouette_scores()
    plot_class.plot_entropy_and_mids_over_years()
    plot_class.plot_entropy_mids_allies_and_lcaprat_over_years()
    plot_class.entropy_and_covariates_over_years()
    plot_class.average_silhouette_score_vs_variables_over_years()
    plot_class.plot_entropy_and_mids_fpca_over_years()

    print('All Well Done! --------------------------')
