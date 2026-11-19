
import os

from gudhi_python_codes.gudhi_plotting_class import GudhiPlottingClass

if __name__ == "__main__":

    beginYear, endYear = 1946, 2010
    years = [1955,1975,1995,2010]
    n_years = [year for year in range(1946, 2011)]
    cases = [n_years.index(year) for year in years]

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    args = {
        'folder_prep': os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/"),
        'Homology': os.path.join(BASE_DIR, "project_data", "Data", "Homology/"),
        'beginYear': beginYear,
        'endYear': endYear,
        'n_comp': 3,
        'years': years,
        'n_years': n_years,
        'cases': cases,
    }

    plot_class = GudhiPlottingClass(args)

    """
    Plotting Persistence Diagrams
    """
    plot_class.Plotting_Persistence_Diagrams()

    """
    Plotting Pairwise Distances between Persistence Diagrams
    """
    plot_class.Plotting_PDs_Pairwise_Distances()
    plot_class.Saving_Dendrogram()
    plot_class.silhouette_score()
    
    """
    Plotting Persistence Barcodes
    """
    plot_class.Plotting_Persistence_Barcodes()


    """
    Plotting entropy
    """
    plot_class.Plotting_Persistence_Entropy()

   
