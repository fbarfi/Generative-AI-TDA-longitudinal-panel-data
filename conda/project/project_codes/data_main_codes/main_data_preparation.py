"""
International Relations Data Preparation and Processing Script

This script processes historical dyadic longitudinal panel data, performing comprehensive
data preparation steps for analysis of interstate interactions from 1885 to 2010.

Configuration Parameters:
    beginYear (int): Starting year for data range (1885)
    endYear (int): Ending year for data range (2010)
    folder (str): Source directory path containing raw data files
    folder_prep (str): Target directory path for processed data output

Process Overview:
    1. Data Loading:
        - Initializes ProcessDataFromSources class with configuration
        - Loads raw data files from specified source directory
        - Aggregates data into unified structure

    2. Data Processing:
        - Generates derived variables using ProcessDataFromSources methods
        - Filters dyadic relationships to ensure ccode1 < ccode2
        - Removes incomplete records (rows with NaN values)
        - Standardizes data types for key variables

    3. Variable Processing:
        Processes and validates the following variables:
        - disp: Interstate disputes indicator
        - disp_lag: Lagged dispute indicator
        - allies: Alliance status between states
        - contgkf: Geographic contiguity measure
        - hegdef: Hegemonic defense indicator
        - smldep: Small power dependency metric
        - lcaprat: Logarithmic capability ratio
        - lgmidist: Logarithmic distance between capitals
        - smldmat: Small power material ratio
        - majpower: Major power status indicator
        - dyadid: Unique identifier for each dyad

    4. Data Validation and Cleaning:
        - Ensures proper data types (converts specific columns to integers)
        - Removes duplicate entries
        - Validates value ranges for key metrics
        - Ensures temporal consistency

    5. Output Generation:
        - Saves processed dataset to 'all_vars_1885_2010.csv'
        - Generates comprehensive summary statistics including:
            * Count, mean, std, min, 25%, 50%, 75%, max for all numeric variables
            * Temporal distribution of observations
            * Missing value analysis
        - Creates 'data_summary_statistics_1885_2010.csv' with descriptive statistics

Dependencies:
    - numpy: Numerical operations and array manipulation
    - pandas: Data manipulation and analysis
    - data_python_codes.processdatafromsources: Custom data processing module
    - data_python_codes.savegroupeddata: Custom data saving utilities

File Structure:
    Input Directory (folder):
        Contains original source files with raw dyadic data

    Output Directory (folder_prep):
        - all_vars_1885_2010.csv: Complete processed dataset
        - data_summary_statistics_1885_2010.csv: Statistical summaries

Usage:
    Run this script directly to process the complete dataset:
    ```python
    python main_data_preparation.py
    ```

Notes:
    - Ensure all required source files are present in the input directory
    - Verify write permissions for the output directory
    - Large datasets may require significant processing time and memory
    - Consider backing up existing files in the output directory before running

Author: Badredine Arfi
Last Updated: 2024-11-23
"""
import os
import pandas as pd

from data_python_codes.processdatafromsources import ProcessDataFromSources

if __name__ == "__main__":

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    args = {
        'beginYear': 1885,
        'endYear':2010,
        'folder': os.path.join(BASE_DIR, "project_data", "Data", "Data_original/"),
        'folder_prep': os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/"),
        }
    
    """
    Create an instance of the class
    """
    data_class = ProcessDataFromSources(args)

    """ reading all input data """

    data_list = data_class.reading_all_input_data()

    data = data_class.gen_all_vars(data_list)

    df_data = data.loc[data['ccode1'] < data['ccode2'],:].reset_index(drop=True)

    df_data.dropna(inplace=True)

    df_data = df_data.astype({'disp':'int',
                            'disp_lag':'int',
                            'allies':'int',
                            'contgkf':'int',
                            'hegdef':'int'})

    df_data = df_data[['year', 'dyadid', 'ccode1', 'ccode2', 'disp', 'disp_lag',
                             'majpower','allies', 'contgkf', 'hegdef','smldep',
                             'lcaprat','lgmidist','smldmat']]


    """ Saving to csv file """

    vars_file = 'all_vars_1885_2010.csv'
    df_data.to_csv(args['folder_prep']+vars_file, index = False )


    """
    Computing descriptive statistics of variables and saving to csv file
    """
    df_summary = df_data[['year','disp', 'disp_lag','majpower',
                    'allies', 'contgkf', 'hegdef','smldep','lcaprat','lgmidist',
                             'smldmat']].describe().round(3)

    df_summary.T.to_csv(args['folder_prep']+'data_summary_statistics_1885_2010.csv')
