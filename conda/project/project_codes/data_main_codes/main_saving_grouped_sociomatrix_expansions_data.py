"""
Sociomatrix Grouping and Expansion Data Processing Script

A comprehensive data processing script that handles the grouping, expansion, and storage of
sociomatrix data for longitudinal panel data analysis, covering the period 1946-2010. The script
specializes in transforming dyadic relationship data into grouped matrix formats with derived features.

Configuration Parameters:
    beginYear (int): Starting year of analysis period (1946)
    endYear (int): Ending year of analysis period (2010)
    data_original (str): Source directory path for raw data files
    data_prepared (str): Target directory path for processed outputs
    n_years (list): Auto-generated list of years from beginYear to endYear, inclusive

Data Processing Workflow:
    1. Matrix Generation and Expansion:
        a) Basic Sociomatrix Creation:
           - Converts dyadic data to matrix format
           - Ensures symmetry where appropriate
           - Handles missing relationships
           - Validates matrix dimensions

        b) Matrix Expansion Operations:
           - Derives additional features from base relationships
           - Computes higher-order network metrics
           - Generates temporal aggregations
           - Calculates interaction effects

    2. Grouping Operations:
        a) Temporal Grouping:
           - Annual aggregations
           - Moving window calculations
           - Period-specific summaries
           - Trend analysis groupings

        b) Structural Grouping:
           - Dyadic relationship clusters
           - Regional groupings
           - Power-based categorizations
           - Behavioral similarity groups

        c) Feature-based Grouping:
           - Relationship intensity levels
           - Interaction type categories
           - Stability-based groups
           - Complex interaction patterns

    3. Data Organization and Storage:
        a) File Structure:
           - Hierarchical organization by year
           - Separate directories for different grouping schemes
           - Clear naming conventions
           - Version control support

        b) Data Formats:
           - Matrix market format for sparse data
           - CSV for dense matrices
           - JSON for metadata
           - HDF5 for large datasets

        c) Quality Control:
           - Checksum generation
           - Format validation
           - Consistency checks
           - Error logging

Output Specifications:
    1. Directory Structure:
        data_prepared/
        ├── annual_matrices/
        │   ├── YYYY_base_matrix.csv
        │   ├── YYYY_expanded_features.csv
        │   └── YYYY_metadata.json
        ├── grouped_data/
        │   ├── temporal_groups/
        │   ├── structural_groups/
        │   └── feature_groups/
        └── summaries/
            ├── group_statistics.csv
            └── processing_log.txt

    2. File Formats:
        - Matrix Files: CSV or sparse formats
        - Metadata: JSON format
        - Summaries: CSV format
        - Logs: Text format

Processing Features:
    1. Data Validation:
        - Matrix dimensionality checks
        - Symmetry validation
        - Value range verification
        - Temporal consistency checks

    2. Expansion Features:
        - Network centrality metrics
        - Path-based calculations
        - Structural equivalence measures
        - Temporal stability indicators

    3. Grouping Criteria:
        - Temporal patterns
        - Structural similarities
        - Behavioral characteristics
        - Interaction intensities

Dependencies:
    Required Packages:
        - data_python_codes.savegroupeddata: Custom module for data saving operations
        - numpy: Numerical operations
        - pandas: Data manipulation
        - scipy.sparse: Sparse matrix operations (if applicable)

    Data Requirements:
        - Preprocessed sociomatrix data
        - Country/entity reference data
        - Temporal indicators
        - Grouping criteria definitions

Usage:
    Basic Execution:
    ```python
    python main_saving_grouped_sociomatrix_expansions_data.py
    ```

Performance Optimization:
    - Implements sparse matrix operations where applicable
    - Uses memory-efficient grouping operations
    - Supports parallel processing for large datasets
    - Implements efficient I/O operations

Error Handling:
    - Validates input data completeness
    - Verifies matrix properties
    - Handles missing or incomplete data
    - Logs processing errors
    - Implements data recovery procedures

Best Practices:
    - Regular backup of output directory
    - Validation of group assignments
    - Documentation of grouping criteria
    - Monitoring of processing resources
    - Regular verification of output integrity

Notes:
    - Process is memory-intensive for large matrices
    - Consider temporal partitioning for extensive datasets
    - Validate grouping criteria before processing
    - Monitor disk space during operation
    - Verify output consistency after processing

Author: Badredine Arfi
Last Updated: 2024-10-23
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
import os

from data_python_codes.savegroupeddata import SaveGroupedData

if __name__ == "__main__":

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    args = {
        'beginYear': 1946,
        'endYear':2010,
         'scaler': scaler,
        'folder': os.path.join(BASE_DIR, "project_data", "Data", "Data_original/"),
        'folder_prep': os.path.join(BASE_DIR, "project_data", "Data", "Data_prepared/"),
        }
    args['n_years'] = [year for year in range(args['beginYear'], args['endYear']+1)]


    save_class = SaveGroupedData(args)
    save_class.save_grouped_data()

    print('Done saving all data to disk')
