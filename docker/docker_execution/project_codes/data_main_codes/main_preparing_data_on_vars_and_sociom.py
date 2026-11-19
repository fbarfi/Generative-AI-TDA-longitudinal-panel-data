""""
Dyadic Data and Sociomatrix Preparation Script for Longitudinal Panel Data Analysis

This script manages the preparation of dyadic longitudinal panel data and sociomatrices,
processing bilateral relationship data between countries from 1946 to 2010. It handles both
the initial data preparation and the optional computation of sociomatrices for network analysis.

Configuration Parameters:
    beginYear (int): Starting year for analysis period (1946)
    endYear (int): Ending year for analysis period (2010)
    data_original (str): Source directory path containing raw data files
    data_prepared (str): Target directory path for processed output
    n_years (list): Auto-generated list of years from beginYear to endYear, inclusive

Detailed Process Flow:
    1. Data Initialization:
        - Sets up configuration parameters
        - Validates input directories and files
        - Generates year list for temporal processing

    2. Dyadic Data Preparation (PreparingDyadicData class):
        a) Data Loading:
           - Reads raw dyadic data files
           - Validates data structure and completeness
           - Handles missing or malformed data

        b) Data Processing:
           - Cleans and standardizes input data
           - Applies necessary transformations
           - Handles temporal alignment of records
           - Manages country code consistency
           - Validates bilateral relationship entries

        c) Quality Control:
           - Ensures data consistency across years
           - Validates relationship symmetry where applicable
           - Handles missing or incomplete relationships
           - Performs temporal consistency checks

    3. Sociomatrix Computation (ComputeSociomatrix class) [Currently Disabled]:
        When enabled, performs the following:
        a) Matrix Construction:
           - Converts dyadic data to matrix format
           - Creates symmetrical matrices where appropriate
           - Handles diagonal elements appropriately

        b) Network Analysis Preparation:
           - Generates adjacency matrices
           - Computes network metrics
           - Prepares data for social network analysis

        c) Temporal Analysis:
           - Creates year-by-year sociomatrices
           - Handles temporal evolution of networks
           - Manages changing network composition

Output Structure:
    data_prepared directory:
        - Processed dyadic datasets
        - Temporal relationship files
        - Network analysis matrices (when sociomatrix computation is enabled)
        - Quality control reports
        - Processing logs and summaries

Data Specifications:
    1. Input Requirements:
       - Dyadic relationship data
       - Country code mappings
       - Temporal coverage indicators
       - Relationship type definitions

    2. Output Formats:
       - Cleaned dyadic datasets
       - Standardized relationship matrices
       - Network analysis ready data
       - Quality assurance reports

Dependencies:
    Required Packages:
        - data_python_codes.preparingdyadicdata: Custom module for dyadic data processing
        - data_python_codes.computesociomatrix: Optional module for sociomatrix computation

    Data Dependencies:
        - Raw dyadic relationship data files
        - Country code reference files
        - Temporal coverage indicators

Usage Examples:
    1. Basic Data Preparation:
        ```python
        python main_preparing_data_on_vars_and_sociom.py
        ```

    2. With Sociomatrix Computation (requires uncommenting):
        - Uncomment the ComputeSociomatrix import
        - Add sociomatrix computation code after dyadic data preparation

Error Handling:
    - Validates input data presence and format
    - Checks for directory permissions
    - Verifies temporal consistency
    - Monitors data quality during processing
    - Logs processing errors and warnings

Performance Considerations:
    - Memory usage scales with number of countries and years
    - Sociomatrix computation can be memory-intensive
    - Consider batch processing for large datasets
    - Temporal partitioning may be necessary for extensive analyses

Notes:
    - Ensure sufficient disk space for output files
    - Back up existing data before processing
    - Monitor system resources during large-scale processing
    - Consider implementing progress tracking for long operations
    - Validate output data before proceeding with analysis

Author: Badredine Arfi
Last Updated: 2024-11-23
"""

""" Preparing the empirical data for using in the analysis """
import os
from data_python_codes.preparingdyadicdata import PreparingDyadicData

if __name__ == "__main__":

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    args = {
        'beginYear': 1946,
        'endYear':2010,
        'folder': os.path.join(BASE_DIR, "project_data", "Data", "Data_original/"),
        'folder_prep': os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/"),
        }
    
    args['n_years'] = [year for year in range(args['beginYear'], args['endYear']+1)]

    """
    Prepraring auxiliary input data and computing sociomatrices
    """
    prep_dyads = PreparingDyadicData(args)
    prep_data = prep_dyads.preparing_data()


    print('All Done')
