import pandas as pd

class ClassData():

    def __init__(self, args):
        super(ClassData, self).__init__()

        self.args = args
        self.nepochs = args['nepochs']
        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.var_names = args['var_names']
        self.vars = args['vars']
        self.n_years = args['n_years']
        self.scaler = args['scaler']
        self.subgraphx_folder = args['subgraphx_folder']
        self.folder_prepared = args['folder_prepared'] 

        # Load and set the GraphX data (train set)
        self.Xij_train, self.yi_train = self._load_data("GraphX")

        # Load and set the Empirical data (test set)
        self.Xij_test, self.yi_test = self._load_data("Empirical")

    def _load_data(self, id_data):
        nepochs = self.nepochs
        scaler = self.scaler
        n_years = self.n_years

        X_list, y_list = [], []

        # Iterate through all years specified in n_years
        for year in n_years:
            usecols = ['dyadid', 'disp', 'majpower', 'allies', 'contgkf', 'hegdef',
                       'smldep', 'lcaprat', 'lgmidist', 'smldmat']
            graph_path = self.subgraphx_folder + f'Reconstructed_data/H_1_SubGraphX_important_data_{year}'
            emp_path = self.folder_prepared + f'dyads_DV&IDVs_pol_igo/dyad_DV&IDVs_pol_igo_{year}'

            # Read dyad IDs for filtering
            emp_dyadids = pd.read_csv(emp_path + '.csv', usecols=['dyadid']).dyadid
            graph_dyadids = pd.read_csv(graph_path + '.csv', usecols=['dyadid']).dyadid

            if id_data == "GraphX":
                # Load GraphX data
                Xy_df = pd.read_csv(graph_path + '.csv',
                                    usecols=usecols).drop('dyadid', axis=1).reset_index(drop=True)

            elif id_data == "Empirical":
                # Load Empirical data, excluding dyads present in GraphX data
                emp_df = pd.read_csv(emp_path + '.csv', usecols=usecols + ['dyadid'])
                Xy_df = emp_df.loc[~emp_df['dyadid'].isin(graph_dyadids)].drop('dyadid', axis=1).reset_index(drop=True)

            y_df = Xy_df['disp']
            X_df = Xy_df[usecols[2:]]

            # Split features into float and int for scaling
            X_float = X_df[['smldep', 'lcaprat', 'lgmidist', 'smldmat']]
            X_int = X_df[['majpower', 'allies', 'contgkf']]

            if len(X_float) > 0:
                # Scale float features and combine with int features
                X_float = pd.DataFrame(data=scaler.fit_transform(X_float.values),
                                       columns=X_float.columns.values).reset_index(drop=True)
                X = pd.concat([X_int, X_float], axis=1)

                X_list.append(X)
                y_list.append(y_df)

        # Combine data from all years
        Xij, yi = self._combine_data(X_list, y_list)
        return Xij, yi

    def _combine_data(self, X_list, y_list):
        Xij = pd.concat(X_list, axis=0).values
        yi = pd.concat(y_list, axis=0).values.reshape((-1,))
        return Xij, yi
