import pymc as pm
import numpy as np
import pytensor.tensor as pt
import arviz as az

from .class_data import ClassData

class BayesianModel():

    def __init__(self, args):
        super(BayesianModel, self).__init__()

        self.args = args
        self.var_names = args['var_names']
        self.vars = args['vars']
        self.folder = args['folder']

        self.data_class = ClassData(args)
        self.Xij_train, self.yi_train = self.data_class.Xij_train, self.data_class.yi_train
        self.Xij_test, self.yi_test = self.data_class.Xij_test, self.data_class.yi_test
        # self.bare_model = []

    def run_model(self):
        X_train = self.Xij_train
        y_train = self.yi_train

        n_vars = X_train.shape[1]
        N_obs = X_train.shape[0]

        with pm.Model() as bare_model:
            tt_X = pm.Data('tt_X', X_train)
            tt_y = pm.Data('tt_y', y_train)

            β = pm.Normal('β', mu=0., sigma=10., shape=(n_vars,))
            p = pm.Deterministic("p", pt.sigmoid(tt_X @ β))

            ψ = pm.Beta('ψ', 3., 3.)
            pm.ZeroInflatedBinomial('y_pred',
                                    psi=ψ,
                                    n=N_obs,
                                    p=p,
                                    observed=tt_y
                                    )
        
        # self.bare_model = bare_model
        
        with bare_model:
            idata = pm.sample(nuts_sampler='pymc',
                                   **self.args['SAMPLE_KWARGS'],
                                   idata_kwargs=self.args['idata_kwargs'])

        model_graph = pm.model_to_graphviz(bare_model)
        graph_file = self.folder + 'bare_model'
        model_graph.render(filename=graph_file, view=False, format='png')
        # az.to_netcdf(idata, self.folder+'MCMC_traces.nc')

    # def out_of_sample(self):
    #     """
    #     Out-of-sample testing using test data from ClassData
    #     """
        X_test = self.Xij_test
        y_test = self.yi_test
    #     idata = az.from_netcdf(self.folder + 'MCMC_traces.nc')

        with bare_model:
            pm.set_data(
                {"tt_X": X_test, "tt_y": y_test})

            posterior_predictive = pm.sample_posterior_predictive(idata)

            idata.extend(posterior_predictive)

            # saving traces to disk
            file_name = self.folder + f'MCMC_traces.nc'
            az.to_netcdf(idata, file_name)

        print('Done saving  MCMC idata with posterior_predictive samples to disk')

        return 