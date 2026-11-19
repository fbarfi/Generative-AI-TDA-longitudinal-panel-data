
import os
os.environ["DGLBACKEND"] = "pytorch"

from SubGraphX_python_codes.plottingsubgraphx import PlottingSubgraphXClass

if __name__ == "__main__":

    nepochs = 200 #int(input("Enter number of epochs used for SubGraphX training (e.g., 200): "))

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_original = os.path.join(BASE_DIR, "project_data", "Data","Data_original/")
    Homology =  os.path.join(BASE_DIR, "project_data", "Data", "Homology/")
    folder_results =  os.path.join(BASE_DIR, "project_data", "Results", "SubGraphX_results/")

    specific_years = [1955,1975,1995,2010]
    h_dim = 'H_1'
    first_year = 1946
    last_year = 2010

    args = {
        'first_year': first_year,
        'last_year': last_year,
        'h_dim': h_dim,
        'data_original': data_original,
        'folder_results': folder_results,
        'Homology': Homology,
        'nepochs': nepochs,
        'specific_years': specific_years,
        }
    args['nyears'] = list(range(args["first_year"],args["last_year"] +1))

    plot_graph = PlottingSubgraphXClass(args)

    plot_graph.plot_subgraph()
