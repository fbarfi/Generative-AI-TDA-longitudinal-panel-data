
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
import matplotlib.pyplot as plt
import arviz as az

from .class_data import ClassData

class SaveResults():

    def __init__(self, args):
        super(SaveResults, self).__init__()

        self.args = args
        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.var_names = args['var_names']
        self.vars = args['vars']
        self.folder = args['folder']
        self.idata = az.from_netcdf(self.folder + 'MCMC_traces.nc')

        self.data_class = ClassData(args)
        self.yi_test = self.data_class.yi_test

    def plot_fig(self, file_name):
        plt.savefig(file_name, dpi=300)

    def plot_save_func(self):
        folder = self.folder
        var_names = self.var_names

        # Trace plots
        az.plot_trace(self.idata, var_names=var_names, compact=False, combined=False,
                      divergences=None, legend=True)
        plt.tight_layout()
        file_trace = folder + 'trace_plots.png'
        self.plot_fig(file_trace)

        # Forest plots
        az.plot_forest(self.idata, var_names=var_names, combined=True, r_hat=True)
        plt.tight_layout()
        file_forest = folder + 'forest_plots.png'
        self.plot_fig(file_forest)

        # Posterior plots
        az.plot_posterior(self.idata, var_names=var_names)
        plt.tight_layout()
        file_posterior = folder + 'posterior_plots.png'
        self.plot_fig(file_posterior)

        # Autocorrelation plots
        az.plot_autocorr(self.idata, var_names=var_names, combined=True)
        plt.tight_layout()
        file_autocorr = folder + 'autocorr_plots.png'
        self.plot_fig(file_autocorr)

        # Density plots
        az.plot_density(self.idata, var_names=var_names)
        plt.tight_layout()
        file_density = folder + 'density_plots.png'
        self.plot_fig(file_density)

        # Energy plots
        az.plot_energy(self.idata)
        plt.tight_layout()
        file_energy = folder + 'energy_plots.png'
        self.plot_fig(file_energy)

    def plot_test_roc_auc(self):
        var_names = self.var_names
        vars = self.vars

        """
        Compute the point prediction by taking the mean and defining
        the category via a threshold
        """
        data = self.idata
        yi_test = self.yi_test
        folder = self.folder

        p_test_pred = data.posterior_predictive["y_pred"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int").to_numpy()

        accuracy = np.mean(yi_test==y_test_pred)

        fpr, tpr, thresholds = roc_curve(
            y_true=yi_test, y_score=p_test_pred, pos_label=1, drop_intermediate=False
        )
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc_display = roc_display.plot(ax=ax, marker="o", markersize=4)
        ax.set_title(
         f"ROC Curve using  Empirical Data as out-of Sample Data\n Accuracy = {accuracy: 0.3f}");
        file_roc  = folder + f'Empirical_data_roc_curve_plot_testing_data.png'

        self.plot_fig(file_roc)

        print("Done plotting and saving mcmc results")

        sum_vars = az.summary(data, var_names = var_names,skipna=True,
                              round_to=4).set_axis(vars,axis=0)

        file_vars = folder + 'vars_summary.csv'
        sum_vars[['mean', 'sd', 'hdi_3%', 'hdi_97%','r_hat']].to_csv(file_vars)

        data_df = sum_vars.reset_index().rename({'index':'Variables'},axis=1)
        data_df[['Variables', 'mean', 'sd', 'hdi_3%',
                 'hdi_97%','r_hat']].set_index('Variables').to_latex(
              buf= folder + f'vars_summary.tex',
             index= True,bold_rows = True,float_format="%.3f", label = 'graphx_generated_data',
            caption ='Bayesian Estimates with GraphX Generated Data')

        print("Done saving MCMC results")

        return
