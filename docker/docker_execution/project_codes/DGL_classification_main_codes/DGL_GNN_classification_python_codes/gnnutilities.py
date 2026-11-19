import os 

from .plotting import PlottingClass

class GNNUtilities():

    def __init__(self, args, data_module):
        self.args = args
        self.data_module = data_module

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Dir '{path}' was created.")
        else:
            print(f"Dir '{path}' already exists.")

    def create_results_folders(self):
        base_folder = self.args['folder']
        self.create_directory(base_folder)
        subfolders = [
            'plots/',
            'GNN_models/',
        ]
        for subfolder in subfolders:
            folder_path = os.path.join(base_folder, subfolder)
            self.create_directory(folder_path)

    def run_plotting(self):
        
        plotting_class = PlottingClass(self.args)

        plotting_class.plot_losses() 
        plotting_class.plot_histograms()
        plotting_class.comparing_actual_predicted_classes()