
import numpy as np
import os
import pandas as pd

import arviz as az
import matplotlib.pyplot as plt

from .runbayesianmodelclass import RunBayesianModelClass

class ObtainResults():

    def __init__(self,args):
        
        super(ObtainResults,self).__init__() 

        self.args = args
        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.var_names = args['var_names']
        self.vars = args['vars']
        self.folder = args['folder']

    def get_idata(self):
       
        var_names = self.var_names
        vars = self.vars
        
        idata = RunBayesianModelClass(self.args).run_model()

        return idata
        
    def plot_fig(self, file_name):
        return plt.savefig(file_name, dpi =300)
                
    def plot_save_func(self):
        
        folder = self.folder
        var_names = self.var_names
        vars = self.vars
        
        data = self.get_idata()
        
        az.plot_trace(data,var_names = var_names,compact=False,combined=False, 
                      divergences=None,legend=True)
        file_trace = folder+'trace_plots.png'
        self.plot_fig(file_trace)
        plt.tight_layout();
        plt.close()
        
        az.plot_forest(data,var_names = var_names,combined=True,r_hat=True)
        plt.tight_layout()
        file_forest = folder + 'forest_plots.png'
        self.plot_fig(file_forest);
        plt.close()
    
        az.plot_posterior(data, var_names = var_names)
        plt.tight_layout()
        file_posterior = folder + 'posterior_plots.png'
        self.plot_fig(file_posterior);
        plt.close()
        
        az.plot_autocorr(data, var_names = var_names,combined=True)
        plt.tight_layout();
        file_autocorr = folder + 'autocorr_plots.png'
        self.plot_fig(file_autocorr);
        plt.close()
            
        az.plot_density(data, var_names =var_names)
        plt.tight_layout();
        file_density  = folder + 'density_plots.png'
        self.plot_fig(file_density)
        plt.close()
        
        az.plot_energy(data)
        plt.tight_layout();
        file_energy  = folder + 'energy_plots.png'
        self.plot_fig(file_energy)
        plt.close()

        print(f"Done plotting and saving mcmc results for {self.args['model_type']} model")
        
        sum_vars = az.summary(data, var_names = var_names,skipna=True,
                              round_to=4).set_axis(vars,axis=0)
        file_vars = folder + 'vars_summary.csv' 
        sum_vars.to_csv(file_vars)

        data_df = sum_vars.reset_index().rename({'index':'Variables'},axis=1)
        data_df[['Variables', 'mean', 'sd', 'hdi_3%', 'hdi_97%',
                 'r_hat']].set_axis(self.vars,axis=0).to_latex(
                    buf= folder + f'vars_summary.tex',
                    index= True,bold_rows = True,float_format="%.3f", 
                    label = 'empirical_data',
                    caption =f'Bayesian Estimates with Empirical Data'
                )

        return
    