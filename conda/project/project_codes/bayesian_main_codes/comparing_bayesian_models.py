

import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BayesianModelComparison:
    """
    Compare different Bayesian models applied to empirical data.
    Uses simplified folder names without scaling method or iteration parameters.
    """

    MODEL_TYPES = {
        'bare': 'Empirical',
        'sociomatrix': 'Empirical_Sociomatrix',
        'sociomatrix_homology': 'Empirical_Sociomatrix_Homology',
        'sociomatrix_homology_latent': 'Empirical_Sociomatrix_Homology_latent',
    }

    def __init__(
        self,
        base_path: Optional[str] = None
    ):
        """
        Initialize the model comparison object.

        Args:
            base_path: Base path to the results directory
        """
        self.BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.base_path = base_path or os.path.join(self.BASE_DIR, "project_data","Results", "Bayesian_results")

        self.df_models = None
        self.compare_dict = None

        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Results directory not found: {self.base_path}")

        logger.info("Initialized BayesianModelComparison")

    def _get_model_path(self, model_type: str) -> str:
        """Generate full path for a model type using basic folder names."""
        filename = f'MCMC_{model_type}'
        path = os.path.join(self.base_path, filename, 'MCMC_traces.nc')
        logger.info(f"Looking for trace file at: {path}")
        return path

    def load_data(self) -> 'BayesianModelComparison':
        """
        Load all model data from NetCDF files.

        Returns:
            self for method chaining
        """
        logger.info("Loading model data...")
        self.compare_dict = {}

        try:
            for model_key, display_name in self.MODEL_TYPES.items():
                path = self._get_model_path(model_key)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"MCMC trace file not found: {path}")

                logger.debug(f"Loading model from {path}")
                self.compare_dict[display_name] = az.from_netcdf(path)

            logger.info("Successfully loaded all model data")
            return self

        except Exception as e:
            logger.error(f"Error loading MCMC traces: {str(e)}")
            raise RuntimeError(f"Error loading MCMC traces: {str(e)}")

    def compare_models(
        self,
        alpha: float = 0.8,
        seed: int = 41
    ) -> 'BayesianModelComparison':
        """
        Compare models using LOO cross-validation.

        Args:
            alpha: Significance level for comparison
            seed: Random seed for reproducibility

        Returns:
            self for method chaining
        """
        if not self.compare_dict:
            raise ValueError("No model data loaded. Run load_data() first")

        logger.info("Comparing models...")
        try:
            self.df_models = az.compare(
                compare_dict=self.compare_dict,
                ic='loo',
                method='stacking',
                alpha=alpha,
                seed=seed,
                scale='negative_log'
            )
            logger.info("Model comparison completed")
            return self

        except Exception as e:
            logger.error(f"Error during model comparison: {str(e)}")
            raise

    def save_table(self) -> 'BayesianModelComparison':
        """
        Save comparison results as LaTeX table.

        Returns:
            self for method chaining
        """
        if self.df_models is None:
            raise ValueError("No comparison results. Run compare_models() first")

        logger.info("Saving comparison table...")
        try:
            data_df = self.df_models.reset_index().rename({'index': 'Models'}, axis=1)
            columns = ['Models', 'rank', 'elpd_loo', 'p_loo', 'elpd_diff', 'weight', 'se']

            output_path = os.path.join(
                self.base_path,
                'comparing_bare_bayesian_model_with_socio_hom_lat.tex'
            )

            data_df[columns].set_index('Models').to_latex(
                buf=output_path,
                index=True,
                bold_rows=True,
                float_format="%.3f",
                label='comparing_models',
                caption='Comparing Bayesian Models:\\ Empirical, Sociomatrix, Homology, and Latent using ELPD'
            )

            logger.info(f"Table saved to {output_path}")
            return self

        except Exception as e:
            logger.error(f"Error saving comparison table: {str(e)}")
            raise

    def plot_comparison(self) -> 'BayesianModelComparison':
        """
        Create and save comparison plot.

        Returns:
            self for method chaining
        """
        if self.df_models is None:
            raise ValueError("No comparison results. Run compare_models() first")

        logger.info("Creating comparison plot...")
        try:
            plt.figure(figsize=(12, 5))
            az.plot_compare(self.df_models, insample_dev=True, textsize=13)
            plt.tight_layout()

            output_path = os.path.join(
                self.base_path,
                'comparative_plot_bare_with_socio_hom_latent_model.png'
            )
            plt.savefig(output_path)
            logger.info(f"Plot saved to {output_path}")

        except Exception as e:
            logger.error(f"Error creating comparison plot: {str(e)}")
            raise

        finally:
            plt.close('all')

        return self

def main() -> int:
    """Main execution function."""
    try:
        logger.info("Starting Bayesian model comparison")

        comparison = BayesianModelComparison()

        (comparison
         .load_data()
         .compare_models()
         .save_table()
         .plot_comparison())

        logger.info("Model comparison completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
