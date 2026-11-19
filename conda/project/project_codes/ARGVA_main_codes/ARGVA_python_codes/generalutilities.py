

import os

class GeneralUtilities:
    """Utility class for handling folder creation and file organization."""

    def __init__(self, args):
        self.args = args
        self.n_epochs = args.get('n_epochs', 0)  # Default to 0 if not provided
        self.folder = args.get('folder', 'results')  # Default value
        self.models_folder = args.get('models_folder', 'models')  # Default value
        self.subfolders = args.get('subfolders', [])  # Default to empty list

    @staticmethod
    def create_directory(path):
        """Create a directory if it does not exist."""
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Dir '{path}' is ready.")
        except OSError as e:
            print(f"Error creating directory '{path}': {e}")

    def create_results_folders(self):
        """Create base results folder and its subdirectories."""
        self.create_directory(self.folder)

        for subfolder in self.subfolders:
            folder_path = os.path.join(self.folder, subfolder)
            self.create_directory(folder_path)
