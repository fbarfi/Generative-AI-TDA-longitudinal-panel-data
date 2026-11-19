import os 

class GNNUtilities():

    def __init__(self, args):
        self.args = args

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Dir '{path}' was created.")
        else:
            print(f"Dir '{path}' already exists.")

    def create_results_folders(self):
        base_folder = self.args['folder_results']
        self.create_directory(base_folder)
        subfolders = [
            'SubGraphX/',
            'Reconstructed_data/',
        ]
        for subfolder in subfolders:
            folder_path = os.path.join(base_folder, subfolder)
            self.create_directory(folder_path)

