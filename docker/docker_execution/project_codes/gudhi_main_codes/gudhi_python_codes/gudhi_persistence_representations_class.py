
import pandas as pd
import numpy as np
import gudhi as gd
from gudhi.representations.vector_methods import Entropy
from gudhi.representations.preprocessing import Padding
# from gudhi.representations import PersistenceImage

from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

from sklearn.preprocessing import MinMaxScaler


class GudhiPersistenceRepresentationsClass():

    def __init__(self,args):

        self.args = args
        self.beginYear  = args['beginYear']
        self.endYear = args['endYear']
        self.n_years = args['n_years']
        self.C = args['C']
        self.p = args['p']
        self.bins = args['bins']
        self.max_edge_length = args['max_edge_length']
        self.max_dimension = args['max_dimension']
        self.n_components = args['n_components']
        self.data_prep = args['folder_prep']
        self.Homology = args['Homology']
        self.scaler = MinMaxScaler()


    """
    Uploading data
    """
    def data_loading(self,year):
        folder = self.data_prep + f'dyads_DV&IDVs_pol_igo/'
        filen = folder + f'dyad_DV&IDVs_pol_igo_{year}.csv'
        data = pd.read_csv(filen)

        Xij = data.drop(['disp'],axis=1)
        yi = data[['dyadid','year','disp']]

        return Xij,yi

    """
    Uploading Rips diagrams
    """
    def uploading_PDs(self):

        PD_list = []
        for year in self.n_years:
            filepd = self.Homology + f'Persistence_Diagrams/PD_hom_H_012_{year}.csv'
            PD_list.append(pd.read_csv(filepd))

        PDs =  [[group[['b', 'd']].to_numpy() for _, group in pd.groupby('h_dim')] for pd in PD_list]
        PDs = [p[1] for p in PDs]
        PDs = Padding().fit_transform(PDs)

        return PDs

    """
    Computing Persistence Images
    """
    def persistence_image(self):
        PDs = self.uploading_PDs()

        PI = gd.representations.PersistenceImage(
            bandwidth=1e-4,
            weight= lambda x: np.arctan(self.C * (x[1])**self.p), 
            resolution=[self.bins, self.bins]
        )

        PI_list = PI.fit_transform(PDs)

        WP = np.array(PI_list).reshape(-1, self.bins, self.bins)
        WP_n = np.zeros_like(WP)

        for i in range(WP.shape[0]):
            WP_n[i, :, :] = self.scaler.fit_transform(WP[i, :, :])
            # scale(WP[i, :, :], axis=0, with_mean=True, with_std=False)

        WP_vec = np.sort(WP_n.reshape(WP.shape[0], -1), axis=-1)
        
        return  WP_vec

    def FPCA_(self):
        WPI = self.persistence_image()
        n_components = self.n_components

        grid_points = np.linspace(0, 1, WPI.shape[1])
        fd = FDataGrid(data_matrix=WPI, grid_points=(grid_points,))

        fpca = FPCA(n_components=n_components)
        fpca_scores = fpca.fit_transform(fd)  # shape: (n_diagrams, n_components)

        df_H = pd.DataFrame(
            data=fpca_scores,          
            columns = [f'H_{i}' for i in range(1,self.n_components +1)]
        )

        df_H.to_csv(
            self.Homology + f'FPCA_hom_H_012_{self.beginYear}_{self.endYear}.csv',
            index=False
        )

        return df_H


    """
    Combining results
    """
    def saving_combined_results(self):

        df_H = self.FPCA_()

        for k,year in enumerate(self.n_years):

            X,y = self.data_loading(year)

            dd = df_H.loc[df_H.index == k].values
            df = pd.DataFrame(data = np.tile(dd,(X.shape[0],1)),
                               columns = df_H.columns.values).reset_index(drop=True)
            X_fin = pd.concat([X.iloc[:,:2],
                               X.iloc[:,range(2,7)],
                               pd.DataFrame(data = self.scaler.fit_transform(X.iloc[:,7:].values),
                                  columns= X.columns[7:])
                              ], axis=1)
            df_comb = pd.concat([ y, X_fin.iloc[:,2:], df], axis=1).reset_index(drop=True)

            df_comb.to_csv(self.Homology + f'IO_data_hom/IO_data_hom_H_012_{year}.csv',
                           index=False)

        return


    """
    Persistence entropy
    """
   
   
    def persistence_entropy(self):

        PE = Entropy()

        PD_list = []
        for year in self.n_years:
            filepd = self.Homology + f'Persistence_Diagrams/PD_hom_H_012_{year}.csv'
            PD_list.append(pd.read_csv(filepd))
     
        PDs_h0, PDs_h1, PDs_h2 = [], [], []
        for df in PD_list:
            for h, PDs_h in zip(
                [0, 1, 2], [ PDs_h0, PDs_h1, PDs_h2]
            ):
                arr = df.loc[df["h_dim"] == h, :][["b", "d"]].values
                PDs_h.append(arr)

        def _entropy(PDs):
            return PE.fit_transform(PDs)
        print(_entropy(PDs_h0))
        
        df = pd.DataFrame({
            "year": self.n_years,
             "H_0": [val[0] for val in _entropy(PDs_h0)],
            "H_1": [val[0] for val in _entropy(PDs_h1)],
            "H_2": [val[0] for val in _entropy(PDs_h2)]
        })

        df.to_csv(self.Homology + f'Entropy_hom_H_012_{self.beginYear}_{self.endYear}.csv', index=False)

        return
       

