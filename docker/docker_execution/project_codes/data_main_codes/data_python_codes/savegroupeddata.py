"""
Parallel Data Processor for Preparing the Data for Bayesian Estimation
=========================================================

This module implements parallel processing for large-scale international relations data analysis,
specifically focused on processing dyadic relationships and IO memberships across multiple years.

The SaveGroupedData class provides efficient parallel computation of spatial network effects
and data transformation using both process-based and thread-based parallelization.

Key Features
-----------
- Multi-year parallel processing using ProcessPoolExecutor
- Threaded matrix operations for spatial power calculations
- Efficient sparse matrix operations for large networks
- Standardized data transformation and storage
- NetCDF output format for analytical data

Requirements
-----------
- Python
- Dependencies:
    - pandas
    - numpy
    - xarray
    - scikit-learn
    - scipy
    - concurrent.futures (standard library)

Input Data Structure
------------------
The code expects two types of input files for each year:

1. Dyadic Variables File:
   - Location: {data_prepared}/dyads_DV&IDVs_pol_igo/dyad_DV&IDVs_pol_igo_{year}.csv
   - Required columns:
     * dyadid: Unique identifier for each dyad
     * year: Year of observation
     * disp: Dependent variable
     * smldep, lcaprat, lgmidist, smldmat: Independent variables
     * majpower, allies, contgkf: Additional categorical variables

2. IO Membership File:
   - Location: {data_prepared}/dyads_IOmembs_pol_igo/dyads_IOmembs_pol_igo_{year}.csv
   - Required columns:
     * row_io_sum: Sum of IO memberships
     * Columns 5 onwards: IO membership indicators

Output Files
-----------
The processor generates the following NetCDF files:
- Xij.nc: Main independent variables matrix
- yi.nc: Dependent variable matrix
- SX1.nc to SX4.nc: Spatial power matrices (1st to 4th order)

Usage Example
-----------
from sklearn.preprocessing import StandardScaler

# Initialize parameters
args = {
    'n_years': range(1990, 2020),
    'scaler': StandardScaler(),
    'data_original': '/path/to/original/data/',
    'data_prepared': '/path/to/prepared/data/'
}

# Create processor instance
processor = SaveGroupedData(args)

# Process and save data
processor.save_grouped_data()

Performance Notes
---------------
1. Year-Level Parallelization:
   - Uses process-based parallelization for independent year processing
   - Automatically utilizes available CPU cores
   - Best performance with multiple years of data

2. Matrix Operations:
   - Uses thread-based parallelization for matrix computations
   - Leverages sparse matrix operations for memory efficiency
   - Optimized for large network matrices

3. Memory Considerations:
   - Processes one year at a time in each process
   - Uses sparse matrices for IO membership calculations
   - Streams data to disk using NetCDF format

Class Methods
------------
__init__(args)
    Initializes the processor with configuration parameters.
    Args:
        args (dict): Configuration dictionary containing:
            - n_years: Range or list of years to process
            - scaler: StandardScaler instance for data normalization
            - data_original: Path to original data directory
            - data_prepared: Path to prepared data directory

compute_spatial_powers(S_sparse, X_values, max_power=4)
    Computes spatial power matrices up to specified order.
    Args:
        S_sparse: Sparse sociomatrix
        X_values: Matrix of independent variables
        max_power: Maximum power to compute (default=4)
    Returns:
        list: List of scaled spatial power matrices

process_io_matrix(df)
    Processes IO membership matrix and computes sociomatrix.
    Args:
        df: DataFrame containing IO membership data
    Returns:
        scipy.sparse.csr_matrix: Sparse sociomatrix

process_single_year(year)
    Processes all data for a single year in parallel.
    Args:
        year: Year to process
    Returns:
        tuple: (X matrix, y vector, list of spatial matrices)

save_grouped_data()
    Main method to process all years and save results.
    Handles parallel processing and data storage.

Error Handling
-------------
- The code assumes input files exist and have correct format
- Missing files will raise FileNotFoundError
- Invalid data format will raise ValueError
- Memory errors may occur with extremely large datasets

Author: Badredine Arfi
Last Updated: October 26, 2024

"""

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

class SaveGroupedData:
    def __init__(self, args):
        self.args = args
        self.n_years = args['n_years']
        self.scaler = args['scaler']
        self.folder = args['folder']
        self.folder_prep = args['folder_prep']

    def compute_spatial_powers(self, S_sparse, X_values, max_power=4):
        """
        Efficiently compute spatial powers and their effect on X
        """
        results = []
        S_power = S_sparse  # Start with S^1

        # Use ThreadPoolExecutor for matrix operations
        with ThreadPoolExecutor() as executor:
            for k in range(1, max_power + 1):
                # Compute S^k * X efficiently using parallel matrix multiplication
                SX = S_power @ X_values

                # Standardize the result (can be done in parallel)
                future = executor.submit(self.scaler.fit_transform, SX)
                SX_scaled = future.result()
                results.append(SX_scaled)

                # Compute next power of S if needed
                if k < max_power:
                    S_power = S_power @ S_sparse

        return results

    def process_io_matrix(self, df):
        """
        Process IO membership matrix efficiently
        """
        # Remove all-zero columns
        df = df.loc[:, ~df.eq(0).all()]

        # Normalize rows in parallel using numpy operations
        row_sums = df['row_io_sum'].astype(float).values
        mask = row_sums != 0
        norm_array = df.iloc[:, 5:].values.astype(float)
        norm_array[mask] = norm_array[mask] / row_sums[mask, np.newaxis]

        # Compute sociomatrix S efficiently using parallel matrix multiplication
        S = norm_array @ norm_array.T
        np.fill_diagonal(S, 0.)

        return sparse.csr_matrix(S)

    def process_single_year(self, year):
        """
        Process data for a single year
        """
        # Load and process main data
        data = pd.read_csv(
            self.folder_prep + f'dyads_DV&IDVs_pol_igo/dyad_DV&IDVs_pol_igo_{year}.csv'
        ).reset_index(drop=True)

        # Process X variables
        X_float = data[['smldep', 'lcaprat', 'lgmidist', 'smldmat']]
        X_float = pd.DataFrame(
            data=self.scaler.fit_transform(X_float.values),
            columns=X_float.columns
        )
        X = pd.concat([
            data[['dyadid', 'year', 'majpower', 'allies', 'contgkf']],
            X_float
        ], axis=1).reset_index(drop=True)

        # Process y variable
        y = data[['dyadid', 'year', 'disp']]

        # Process IO membership data
        df = pd.read_csv(
            self.folder_prep + f'dyads_IOmembs_pol_igo/dyads_IOmembs_pol_igo_{year}.csv'
        )

        # Compute sparse sociomatrix
        S_sparse = self.process_io_matrix(df)

        # Compute all spatial powers and their effects efficiently
        SX_results = self.compute_spatial_powers(
            S_sparse,
            X.values[:, 2:],  # Exclude dyadid and year
            max_power=4
        )

        # Convert SX results to DataFrames
        SX_dfs = [pd.DataFrame(
            data=SX,
            columns=X.columns.values[2:]
        ) for SX in SX_results]

        return X, y, SX_dfs

    def save_grouped_data(self):
        # Process years in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_single_year, self.n_years))

        # Unpack results
        X_list, y_list, SX_lists_nested = zip(*results)

        # Transpose SX_lists_nested to group by power instead of by year
        SX_lists = [[] for _ in range(4)]
        for year_results in SX_lists_nested:
            for power_idx, SX_df in enumerate(year_results):
                SX_lists[power_idx].append(SX_df)

        # Concatenate all results
        Xij = pd.concat(X_list, axis=0).reset_index(drop=True)
        yi = pd.concat(y_list, axis=0).reset_index(drop=True)

        # Save main data
        Xij.to_xarray().to_netcdf(self.folder_prep + 'Xij.nc', mode='w')
        yi.to_xarray().to_netcdf(self.folder_prep + 'yi.nc', mode='w')

        # Save spatial results
        for i, SX_list in enumerate(SX_lists, 1):
            SX = pd.concat(SX_list, axis=0).reset_index(drop=True)
            SX.to_xarray().to_netcdf(
                self.folder_prep + f'SX{i}.nc',
                mode='w'
            )