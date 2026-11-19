# Guide: Running Scripts in Conda Environments

This guide provides step-by-step instructions for running eight Python scripts  (executing the different parts of the paper), each in its own Conda environment. Follow these steps to execute each script properly.

## Prerequisites

- Ensure you have Conda installed on your system. 
- Create all needed conda environments. See the utilities_scripts folder in the main folder for executing this task.
- Beware that the scripts take time to execute and hence it is better to execute them in the background of your machine. Script 6 is the longest one to run. 
- Keep the structures of the various directories as is since they depend on each another for executing the scripts and storing the results. 
- Keep the execution order as given down below since latter scripts draw on results of earlier ones, with the intermediate results automatically saved where they should be. 
- The various codes produce a lot more results than what is published in the paper and these results are stored in various subdirectories in the folder project_data (see the accompanying md files for the structures of the two main folders - project_codes and project_data)

## General Process

For each script:
1. Activate the appropriate Conda environment in a terminal window
2. Run the script
3. Deactivate the environment

## Script Execution Details

### 0. Go to folder `project_codes/scripts_master/`

### 1. Running 1_run_all_data_scripts.py

```bash
# Activate the Conda environment
conda activate env_1

# Run the script
python 1_run_all_data_scripts.py

# Deactivate the environment
conda deactivate
```

### 2. Running 2_1_run_all_gudhi_scripts.py

```bash
# Activate the Conda environment
conda activate env_2_1

# Run the script
python 2_1_run_all_gudhi_scripts.py

# Deactivate the environment
conda deactivate
```
### 2.2 Running 2_2_run_graph_data_scripts.py

```bash
# Activate the Conda environment
conda activate env_2_2

# Run the script
python 2_2_run_graph_data_scripts.py

# Deactivate the environment
conda deactivate
```


### 3. Running 3_run_all_dgl_classification_scripts.py

```bash
# Activate the Conda environment
conda activate env_3

# Run the script
python 3_run_all_dgl_classification_scripts.py

# Deactivate the environment
conda deactivate
```

### 4. Running 4_run_all_ARGVA_scripts.py

```bash
# Activate the Conda environment
conda activate env_4

# Run the script
python 4_run_all_ARGVA_scripts.py

# Deactivate the environment
conda deactivate
```

### 5. Running 5_run_all_bayesian_scripts.py

```bash
# Activate the Conda environment
conda activate env_5

# Run the script
python 5_run_all_bayesian_scripts.py

# Deactivate the environment
conda deactivate
```

### 6. Running 6_run_all_SubGraphX_scripts.py

```bash
# Activate the Conda environment
conda activate env_6

# Run the script
python 6_run_all_subgraphx_scripts.py

# Deactivate the environment
conda deactivate
```

### 7. Running 7_run_all_bayesian_subgraphx_scripts.py

```bash
# Activate the Conda environment
conda activate env_7

# Run the script
python 7_run_all_bayesian_subgraphx_scripts.py

# Deactivate the environment
conda deactivate
```
### 8. Running 8_extract_paper_figures_and_tables_to_new_folder.py
Collects the figures and tables in the main paper in one folder named `paper_figures_and_tables_extracted_from_results_folder' in the main directory of the project

```bash
# Run the script
python 8_extract_paper_figures_and_tables_to_new_folder.py
```

## Notes

- Ensure that you are in the correct directory containing the scripts before running these commands.
- If you encounter any errors, make sure that the Conda environments are correctly set up with all required dependencies.

## Troubleshooting

If you encounter issues:
1. Verify that the script file exists in the current directory.
2. Check that the Conda environment is properly created and contains all necessary dependencies.
3. Ensure you have the necessary permissions to execute the scripts.

For any persistent issues, consult the error messages and refer to the documentation for the specific libraries or tools used in each script.

### Author: Badredine Arfi
### Date: July 14, 2025


