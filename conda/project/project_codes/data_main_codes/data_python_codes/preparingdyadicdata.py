"""
International Organization (IO) Dyadic Data Preparation Module
=========================================================

This module processes and prepares dyadic data related to international organizations (IOs),
focusing on political IGOs and their relationships with country pairs. It handles data loading,
filtering, and organization for both dyadic and state-level analyses.

Main Functionality
----------------
1. Processes IO membership data for country pairs
2. Combines IO data with other international relations variables
3. Organizes data in both dyadic and state-level formats
4. Creates yearly datasets for analysis
5. Handles political IGO filtering and data cleaning

Data Processing Steps
-------------------
1. IO Information Processing:
   - Loads IO membership data
   - Filters for political IGOs
   - Processes membership patterns

2. Dyadic Information:
   - Creates country-pair relationships
   - Processes membership information
   - Handles missing and invalid values

3. Variables Processing:
   - Combines with other international relations variables
   - Creates state-level and dyadic outputs
   - Ensures temporal consistency

Required Input Data
-----------------
In specified input folder:
    1. list_IOs_COW_V3_short.csv: IO names and information
    2. igo_year_format_v3.csv: IGO membership data
    3. dyadic_format_v3.dta: Dyadic relationships
    4. all_vars_1885_2010.csv: International relations variables

Parameters
----------
args : dict
    Configuration dictionary containing:
    - beginYear: Start year for analysis
    - endYear: End year for analysis
    - n_years: Array of years to process
    - data_original: Path to source data files
    - data_prepared: Path for output files

Output Structure
--------------
Creates four types of output files per year:
1. Dyadic format for Bayesian models:
   - dyads_DV&IDVs_pol_igo_{year}.csv
   - dyads_IOmembs_pol_igo_{year}.csv

2. State-level format for GNN models:
   - states_DV&IDVs_pol_igo_{year}.csv
   - states_IOmembs_pol_igo_{year}.csv
   - states_pol_igo_{year}.csv

Key Methods
----------
io_info():
    Processes IO information and filters political IGOs

dyads_info():
    Processes dyadic relationships and IO memberships

vars_info():
    Processes international relations variables

preparing_data():
    Combines all data and creates output files

Generated Variables
-----------------
Core Variables:
    - year: Year of observation
    - state_a, state_b: Country codes
    - dyadid: Unique identifier for country pair
    - IO membership variables: Binary indicators for each IO
    - row_io_sum: Total IO memberships per dyad

Additional Variables (from vars_info):
    - disp: Disputes
    - disp_lag: Lagged disputes
    - majpower: Major power status
    - allies: Alliance status
    - contgkf: Contiguity
    - hegdef: Hegemonic defense
    - smldep: Small state dependency
    - lcaprat: Log capability ratio
    - lgmidist: Log distance
    - smldmat: Small state material interests

Notes
-----
1. Focuses specifically on political IGOs
2. Handles both dyadic and state-level data structures
3. Creates separate files for different modeling approaches
4. Maintains data consistency across years
5. Filters out inactive or irrelevant IOs

Dependencies
-----------
- numpy: Numerical operations
- pandas: Data manipulation and analysis

Usage Example
------------
```python
args = {
    'beginYear': 1946,
    'endYear': 2010,
    'n_years': np.arange(1946, 2011),
    'data_original': 'data/original/',
    'data_prepared': 'data/prepared/'
}

processor = PreparingDyadicData(args)
processor.preparing_data()
"""

import pandas as pd
import numpy as np

class PreparingDyadicData():

    def __init__(self,args):

        super(PreparingDyadicData,self).__init__()

        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.n_years = args['n_years']
        self.folder = args['folder']
        self.folder_prep = args['folder_prep']

    """
    Importing information on IOS
    """
    def io_info(self):
        folder = self.folder
        n_years = self.n_years

        beginYear = self.beginYear
        endYear = self.endYear

        """ Import data on IOs names """

        file_io_list = folder+'list_IOs_COW_V3_short.csv'
        df_io_list = pd.read_csv(file_io_list)

        """ Importing data and selecting those IOs that are in 'list_IOs_COW_V3_short.csv' only """

        file_io_info = folder+'igo_year_format_v3.csv'
        df_dyad_io_info = pd.read_csv(file_io_info,low_memory=False).drop(columns =
                ['sdate', 'deaddate','dead', 'integrated', 'replaced','longorgname',
                 'version','accuracyofpre1965membershipdates',
                'sourcesandnotes','imputed','social','economic'])

        """ Convert object columns to string """
        obj_cols = df_dyad_io_info.select_dtypes(include=['object']).columns
        df_dyad_io_info[obj_cols] = df_dyad_io_info[obj_cols].astype('string')

        """ Selecting data for beginYear <= year <= endYear """
        df_dyad_io_info = (df_dyad_io_info.loc[(df_dyad_io_info['year'] >= beginYear ) &
                                        (df_dyad_io_info['year'] <= endYear),:])

        common_ios = np.intersect1d(df_io_list.ionum,df_dyad_io_info.ionum)
        df_dyad_io_info = df_dyad_io_info.loc[df_dyad_io_info.ionum.isin(common_ios)].sort_values(by =
                                                                                            ['ionum'])

        """ Kepping only political igos """
        df_dyad_io_info = df_dyad_io_info.loc[df_dyad_io_info['political'] ==1, :].drop(['political'],
                                                                                              axis=1)

        df_dyad_io_info = df_dyad_io_info.sort_values(by=[
                                    'year','ionum']).fillna(0.).reset_index(drop=True)

        return df_dyad_io_info


    def dyads_info(self):
        folder = self.folder
        beginYear = self.beginYear
        endYear = self.endYear

        """
        Import info on observed dyadic memberships in IGOs
        """
        df_dyad_io_info = self.io_info()

        """
        Upload data on dyads and ios
        """
        file_dyad_io = folder+'dyadic_format_v3.dta'
        df_io = pd.read_stata(file_dyad_io)

        """
        Selecting data for year = beginYear to year = endYear
        """
        df_dyad_io = df_io.loc[(df_io['year'] >= beginYear )
                    & (df_io['year'] <= endYear),:].reset_index(drop=True)

        """
        Dropping unneeded variables and constructing dyadid variable,
        then sorting by year and by dyadid
        """
        df_dyad_io = df_dyad_io.drop(['country1','country2','state'], axis = 1)

        df_dyadid = pd.DataFrame({'dyadid':( df_dyad_io.ccode1)*1000. + df_dyad_io.ccode2}).astype(int)

        df_dyad_io = pd.concat([df_dyadid,df_dyad_io],axis=1)
        df_dyad_io = df_dyad_io.sort_values(by = ['year','dyadid']).drop(['ccode1','ccode2'],
                                                                         axis=1).reset_index(drop=True)
        """
        Replacing -9 and -1 values with 0
        """
        df_dyad_io.replace([-9,-1],0,inplace=True)
        all_ios = np.array([word.lower() for word in df_dyad_io.columns.values])[2:]
        names_ar = np.array([word.lower() for word in list(pd.unique(df_dyad_io_info.ioname))])

        df_dyad_io_rel = df_dyad_io.loc[:, np.intersect1d(all_ios,names_ar)]
        relevant_dyadic_ios = pd.concat([df_dyad_io.iloc[:,:2],df_dyad_io_rel],axis=1)
        df_final = relevant_dyadic_ios.reset_index(drop=True)

        return df_final

    def vars_info(self):
        beginYear = self.beginYear
        endYear = self.endYear
        folder =self.folder
        folder_prep = self.folder_prep

        n_years = self.n_years

        """
        Data on Covariates for Major Powers from beginYear 1885 to endYear 2010
        """
        file_vars = folder_prep +'all_vars_1885_2010.csv'
        dyad_vars = pd.read_csv(file_vars)

        dyad_vars = dyad_vars.loc[(dyad_vars['year']>= beginYear)
                          & (dyad_vars['year'] <= endYear),:]
        dyad_years =  pd.DataFrame({'year': pd.unique(dyad_vars.year)})

        """ Selecting common years """
        com_years = pd.DataFrame({'year': np.intersect1d(dyad_years, n_years)})
        dyad_vars = dyad_vars.loc[dyad_vars['year'].isin(com_years.year),:]

        """ Keeping only variables relevant to study """
        df = dyad_vars[['year', 'dyadid', 'disp', 'disp_lag','majpower', 'allies',
                    'contgkf', 'hegdef', 'smldep', 'lcaprat','lgmidist', 'smldmat']]

        df_statea = pd.DataFrame({'state_a': (np.floor(df.dyadid/1000)).astype(int) })
        df_stateb = pd.DataFrame({'state_b':(df.dyadid -
                                             (np.floor(df.dyadid/1000))*1000).astype(int) })

        df_dyad_vars = pd.concat([df_statea,df_stateb,df],axis=1).reset_index(drop=True)

        df_vars = df_dyad_vars[['year', 'state_a','state_b','dyadid', 'disp', 'disp_lag',
                        'majpower', 'allies', 'contgkf', 'hegdef', 'smldep', 'lcaprat',
                                       'lgmidist', 'smldmat']].reset_index(drop=True)

        return df_vars

    def preparing_data(self):

        folder = self.folder
        folder_prep = self.folder_prep
        df_dyad_vars = self.vars_info()
        df_dyadic_ios = self.dyads_info()

        n_years = self.n_years

        """
        Finding common dyads in data on vars and data on ios
        """
        for year in n_years:

            df_dyad_vars_year = df_dyad_vars.loc[df_dyad_vars['year'] == year,:].reset_index(drop=True)
            df_dyad_io_year = df_dyadic_ios.loc[df_dyadic_ios['year'] == year,:].reset_index(drop=True)

            """
            Summing up IO-memberships for each dyad (summing up all row elements)
            """
            df_sum_rows = df_dyad_io_year.iloc[:,5:].sum(axis=1).values

            """
            Creating a column for storing row sums
            """
            df_dyad_io_year.insert(loc = 2, column = 'row_io_sum', value = df_sum_rows)

            """
            Making sure that no row is all zeros (that is, dyad with no memberships)
            """
            ff_df_dyad_io_year = df_dyad_io_year.loc[~(df_dyad_io_year.iloc[:,3:] == 0).all(axis=1),:]

            """
            Finding common dyads in data on ios and data on vars
            """
            df_common_dyads = pd.merge(left = ff_df_dyad_io_year['dyadid'] ,
                                       right =  df_dyad_vars_year['dyadid'],
                                       how ='inner',on = 'dyadid')

            """
            Keeping data on ios with common dyads only
            """
            df_merge = pd.merge(left = ff_df_dyad_io_year,right = df_common_dyads ,
                                how ='inner', on = 'dyadid')
            df_ios = df_merge.reset_index(drop=True)

            df_ios['state_a'] = df_ios['dyadid'].apply( lambda x: x // 1000)
            df_ios['state_b'] = df_ios['dyadid'].apply(lambda x:  x % 1000)

            columns = df_ios.columns.tolist()
            new_order = columns[:1] + columns[-2:] + columns[1:-2]
            df_ios = df_ios[new_order]

            """
            Keeping data on vars with common dyads only
            """
            df_vars_merge = pd.merge(left = df_common_dyads,right = df_dyad_vars_year ,
                                     how ='inner',on ='dyadid')
            df_vars = df_vars_merge.reset_index(drop=True)

            """ Saving results to disk to folder : '../Data/Data_prepared/' """

            """ As dyads first: needed for Bayesian models """
            dyad_vars_file = f'dyads_DV&IDVs_pol_igo/dyad_DV&IDVs_pol_igo_{year}.csv'
            df_vars.to_csv(folder_prep+dyad_vars_file,index=False)

            dyad_file_IOmembs = f'dyads_IOmembs_pol_igo/dyads_IOmembs_pol_igo_{year}.csv'
            df_ios.to_csv(folder_prep+dyad_file_IOmembs, index = False)

            """ As states: needed for GNN models """
            vars_file = f'states_DV&IDVs_pol_igo/states_DV&IDVs_pol_igo_{year}.csv'
            df_vars.to_csv(folder_prep+vars_file,index=False)

            file_IOmembs = f'states_IOmembs_pol_igo/states_IOmembs_pol_igo_{year}.csv'
            df_ios.to_csv(folder_prep+file_IOmembs, index = False)

            df_states = pd.DataFrame(data = df_ios['state_a'].unique().tolist() +
                                             df_ios['state_b'].unique().tolist(),columns = ['state'])
            df_states = df_states.drop_duplicates(keep='first'
                                                 ).sort_values(by = ['state']).reset_index(drop=True)
            file_states = f'states_pol_igos/states_pol_igo_{year}.csv'
            df_states.to_csv(folder_prep+file_states, index = False)

        return