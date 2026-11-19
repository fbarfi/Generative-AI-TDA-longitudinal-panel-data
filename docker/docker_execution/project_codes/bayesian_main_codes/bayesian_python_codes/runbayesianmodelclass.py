import pymc as pm
import arviz as az
import matplotlib.pyplot as plt 

from .bayesianmodelsclass import BayesianModelsClass

class RunBayesianModelClass():

    def __init__(self,args):

        super(RunBayesianModelClass,self).__init__()

        self.args = args
        self.var_names = args['var_names']
        self.vars = args['vars']
        self.n_years = args['n_years']
        self.folder = args['folder']
        self.model_type = args['model_type']
        self.bayes_model = BayesianModelsClass(args)

    def run_model(self):
        args = self.args
        SAMPLE_KWARGS = self.args['SAMPLE_KWARGS']
        var_names = self.var_names
        vars = self.vars
        model = self.bayes_model.construct_model()

        with model:
            idata = pm.sample(**SAMPLE_KWARGS,
                                idata_kwargs = args['idata_kwargs'])

        # if one wants to compute posterior predictive sample
        # with model:
        #     idata.extend(pm.sample_posterior_predictive(idata))
        #     idata.extend(pm.sample_prior_predictive(var_names = var_names ))

        # to save the traces to disk
        file_name = self.folder+f'MCMC_traces.nc'
        with model:
            az.to_netcdf(idata,file_name)

        model_graph = pm.model_to_graphviz(model)
        graph_file = self.folder+f'{self.model_type}_model'
        model_graph.render(filename=graph_file,view = False, format='png')
    

        return idata