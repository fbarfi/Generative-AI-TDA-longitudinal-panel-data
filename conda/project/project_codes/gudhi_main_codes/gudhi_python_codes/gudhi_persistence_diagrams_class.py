import pandas as pd 
import numpy as np
import gudhi as gd
from gudhi.sklearn.rips_persistence import RipsPersistence
from sklearn.preprocessing   import MinMaxScaler


class GudhiPersistenceDiagramsClass():
    def __init__(self,args):
        
        self.args = args
        self.beginYear  = args['beginYear']
        self.endYear = args['endYear']
        self.n_years = args['n_years']
        self.max_edge_length = args['max_edge_length']
        self.max_dimension = args['max_dimension']
        self.vars = ['disp', 'majpower', 'allies', 'contgkf',
                       'smldep', 'lcaprat', 'lgmidist', 'smldmat']
        self.folder_prep = args['folder_prep']
        self.Homology = args['Homology']
        self.homology_dimensions = args['homology_dimensions']
         

    """
    Uploading data 
    """     
    def data_loading(self,year):
        vars = self.vars[1:]
        folder = self.folder_prep
        
        Xij = pd.read_csv(folder+f'dyads_DV&IDVs_pol_igo/dyad_DV&IDVs_pol_igo_{year}.csv',
                              usecols = vars)
        
        iombs = pd.read_csv(
                folder+f'dyads_IOmembs_pol_igo/dyads_IOmembs_pol_igo_{year}.csv',
                usecols =['row_io_sum'])
        iombs = iombs/iombs.max()
        
        X = pd.concat([Xij,iombs],axis=1).values
        
        return X
    
    
    """
    Computing diagrams
    """
    def rips(self, year):
        X = self.data_loading(year)
        rips_pers = RipsPersistence(homology_dimensions = self.homology_dimensions, 
                                    input_type = 'point cloud') 
        diagrams = rips_pers.fit_transform([X])

        # selector = gd.representations.DiagramSelector(limit=np.inf, point_type="finite")
        # diagram = selector.fit_transform(diagrams[0])
        # scaler =    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])
        # diagram = scaler.fit_transform(diagram)

        return diagrams[0]

    def computing_one_PD(self,year):
            
        diagram = self.rips(year)
        d_list = []
        for i,d in enumerate(diagram):
            dDict= {
                'h_dim': np.tile(i,d.shape[0]),
                'b': d[:,0],
                'd': d[:,1],
                    }
            d_list.append(pd.DataFrame(dDict))
        df_H = pd.concat([*d_list],axis=0)

        filepd = self.Homology + f'Persistence_Diagrams/PD_hom_H_012_{year}.csv'
        df_H.to_csv(filepd,index=False)

        return df_H

    def grouping_PDs(self):
        n_years = self.n_years
        
        for year in n_years:
            self.computing_one_PD(year)
        
        return 