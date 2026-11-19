import numpy as np
import pandas as pd
import xarray as xr

class DataClass():
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.var_names = args['var_names']
        self.vars = args['vars']
        self.n_years = args['n_years']
        self.scaler = args['scaler']
        self.folder = args['folder']
        self.data_folder = args['data_folder']
        self.data_prepared = args['data_prepared']
        self.model_type = args['model_type']
        self.n_years = args['n_years']
        self.int_vars = ['year','majpower', 'allies', 'contgkf']
        self.float_vars = ['smldep','lcaprat', 'lgmidist','smldmat']

    def load_sociomatrix_files(self):
        """Helper function to load SX files."""
        SX_files = [
            self.data_prepared + 'SX1.nc',
            self.data_prepared + 'SX2.nc',
            self.data_prepared + 'SX3.nc',
            self.data_prepared + 'SX4.nc',
        ]

        SX_list = []
        for file in SX_files:
            SX = xr.open_dataset(file).to_dataframe()
            SX_list.append(SX.values)

        return SX_list

    def scale_latent_data(self):
        latent_data = pd.read_csv(self.data_prepared + 'augmented_longutidinal_panel_data.csv',
                                      usecols=['year','L0', 'L1'])
        Latent_list = []
        for year in self.n_years:
            latent_data_year = latent_data.loc[latent_data['year']== year].reset_index(drop=True)
            Latent_list.append(pd.DataFrame(data =  self.scaler.fit_transform(latent_data_year[['L0','L1']].values),
                                            columns = ['L0','L1']) )
        scaled_latent_data =pd.concat([*Latent_list],axis=0).reset_index(drop = True)

        return scaled_latent_data

    def get_data(self):

        vars = self.vars
        var_names = self.var_names

        filexij = self.data_folder + 'Data_prepared/Xij.nc'
        fileyi = self.data_folder +  'Data_prepared/yi.nc'

        Xij = xr.open_dataset(filexij).to_dataframe().drop(['dyadid'],  axis=1)
        yi = xr.open_dataset(fileyi).to_dataframe().drop(['dyadid'], axis=1)

        X_data = Xij.drop(['year'],axis=1).values
        y_data = yi['disp'].values

        data_dict = {
                'Xij': X_data,
                'yi': y_data,
            }

        if self.model_type in ['sociomatrix', 'sociomatrix_homology', 'sociomatrix_homology_latent']:
            data_dict['SX'] = self.load_sociomatrix_files()

        if self.model_type == 'sociomatrix_homology':
            hom_data = pd.read_csv(self.data_folder + f'Homology/IO_data_hom_H_012_{self.beginYear}_{self.endYear}.csv',
                                   usecols=['year', 'H_1', 'H_2'])

            data_dict['Hom'] = hom_data.drop(['year'], axis=1).values

        if self.model_type == 'sociomatrix_homology_latent':

            Latent_data = self.scale_latent_data()

            hom_data = pd.read_csv(self.data_folder + f'Homology/IO_data_hom_H_012_{self.beginYear}_{self.endYear}.csv',
                                   usecols=[ 'year','H_1', 'H_2'])

            data_dict['Hom'] = hom_data.drop(['year'], axis=1).values

            data_dict['Latent'] = Latent_data.values

        return data_dict