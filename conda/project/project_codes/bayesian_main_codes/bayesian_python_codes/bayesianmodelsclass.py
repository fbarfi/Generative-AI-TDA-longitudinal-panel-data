from pymc import model_to_graphviz
import pymc as pm
import pytensor.tensor as pt

import numpy as np
import pandas as pd

from .dataclass import DataClass

class BayesianModelsClass():

    def __init__(self,args):

        super(BayesianModelsClass,self).__init__()

        self.args = args
        self.model_type = args['model_type']
        self.data_class = DataClass(args)

    def construct_model(self):
        model_type = self.model_type

        if model_type == 'bare':

            datasDict = self.data_class.get_data()

            X_data = datasDict['Xij']
            y_data = datasDict['yi']

            N_obs, n_vars = X_data.shape

            del datasDict

            with pm.Model() as model:

                tt_X = pm.Data('tt_X', X_data)

                β = pm.Normal('β', mu=0., sigma=10.,shape = (n_vars,))
                η = tt_X @ β
                ψ = pm.Beta('ψ',3.,3.)
                y_pred = pm.ZeroInflatedBinomial('y_pred',
                                     psi = ψ,
                                     n = N_obs,
                                     p = pt.sigmoid(η),
                                     observed = y_data,
                                    )

        elif model_type == 'sociomatrix':

            datasDict = self.data_class.get_data()

            X_data = datasDict['Xij']
            y_data = datasDict['yi']
            SX_list = datasDict['SX']

            del datasDict

            N_obs, n_vars = X_data.shape

            with pm.Model() as model:

                tt_X = pm.Data('tt_X', X_data)

                tt_XS_1 = pm.Data('tt_XS_1', SX_list[0])
                tt_XS_2 = pm.Data('tt_XS_2', SX_list[1])
                tt_XS_3 = pm.Data('tt_XS_3', SX_list[2])
                tt_XS_4 = pm.Data('tt_XS_4', SX_list[3])

                β = pm.Normal('β', mu=0., sigma=10.,shape = (n_vars,))
                ρ = pm.Beta('ρ', 3.,3.)

                η = (tt_X + ρ*tt_XS_1 + ρ**2 *tt_XS_2 + ρ**3 * tt_XS_3 + ρ**4 * tt_XS_4) @ β

                ψ = pm.Beta('ψ',3.,3.)

                y_pred = pm.ZeroInflatedBinomial('y_pred',
                                     psi = ψ,
                                     n = N_obs,
                                     p = pt.sigmoid(η),
                                     observed = y_data)


        elif model_type == 'sociomatrix_homology':

            datasDict = self.data_class.get_data()

            y_data = datasDict['yi']
            X_data = datasDict['Xij']
            SX_list = datasDict['SX']
            H_data = datasDict['Hom']

            del datasDict

            N_obs, n_vars = X_data.shape
            n_H =  H_data.shape[1]

            with pm.Model() as model:

                tt_X = pm.Data('tt_X', X_data)
                tt_H = pm.Data('tt_H', H_data)

                tt_XS_1 = pm.Data('tt_XS_1', SX_list[0])
                tt_XS_2 = pm.Data('tt_XS_2', SX_list[1])
                tt_XS_3 = pm.Data('tt_XS_3', SX_list[2])
                tt_XS_4 = pm.Data('tt_XS_4', SX_list[3])

                β = pm.Normal('β', mu=0., sigma = 1.,shape = (n_vars,))
                ζ = pm.Normal('ζ', mu= 0., sigma =  1. , shape = (n_H,))
                δ = pm.Normal('δ', mu= 0., sigma =  1. , shape = (n_H,))
                ρ = pm.Beta('ρ', 3.,3.)

                γ = pt.sum(tt_X * pt.tile(tt_H * δ[None,:],
                                          (1, (n_vars // n_H) + 1))[:, :n_vars],axis=1)

                η = γ + tt_H @ ζ   + (tt_X + ρ*tt_XS_1 + ρ**2 *tt_XS_2 + ρ**3 * tt_XS_3 + ρ**4 * tt_XS_4) @ β

                ψ = pm.Beta('ψ',3.,3.)
                y_pred = pm.ZeroInflatedBinomial('y_pred',
                                     psi = ψ,
                                     n = N_obs,
                                     p = pt.sigmoid(η),
                                     observed = y_data)

        elif  model_type == 'sociomatrix_homology_latent':

            datasDict = self.data_class.get_data()

            y_data = datasDict['yi']
            X_data = datasDict['Xij']
            SX_list = datasDict['SX']
            H_data = datasDict['Hom']
            L_data = datasDict['Latent']

            del datasDict

            N_obs, n_vars =X_data.shape
            n_H =  H_data.shape[1]
            n_L =  L_data.shape[1]

            with pm.Model() as model:

                tt_X = pm.Data('tt_X', X_data)
                tt_H = pm.Data('tt_H', H_data)
                tt_L = pm.Data('tt_L', L_data)

                tt_XS_1 = pm.Data('tt_XS_1', SX_list[0])
                tt_XS_2 = pm.Data('tt_XS_2', SX_list[1])
                tt_XS_3 = pm.Data('tt_XS_3', SX_list[2])
                tt_XS_4 = pm.Data('tt_XS_4', SX_list[3])

                β = pm.Normal('β', mu=0., sigma = 1.,shape = (n_vars,))

                ζ = pm.Normal('ζ', mu= 0., sigma =  1. , shape = (n_H,))
                δ = pm.Normal('δ', mu= 0., sigma =  1. , shape = (n_H,))

                ξ = pm.Normal('ξ', mu=0., sigma=1., shape=(n_L,))
                θ = pm.Normal('θ', mu=0., sigma=1., shape=(n_L,))

                ρ = pm.Beta('ρ', 3.,3.)

                γ = pt.sum(tt_X * pt.tile(tt_H * δ[None,:],
                                          (1, (n_vars // n_H) + 1))[:, :n_vars], axis=1)

                ω = pt.sum(tt_X * pt.tile(tt_L * θ[None,:],
                                          (1, (n_vars // n_L) + 1))[:, :n_vars], axis=1)

                η = γ + (tt_H @ ζ) + ω + (tt_L @ ξ) + (tt_X + ρ*tt_XS_1 + ρ**2 *tt_XS_2 + ρ**3 * tt_XS_3 + ρ**4 * tt_XS_4) @ β

                ψ = pm.Beta('ψ',3.,3.)
                y_pred = pm.ZeroInflatedBinomial('y_pred',
                                     psi = ψ,
                                     n = N_obs,
                                     p = pt.sigmoid(η),
                                     observed = y_data)


        print(model.debug())
        return model