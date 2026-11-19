import os
import shutil
from pathlib import Path

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"[❌] Failed to create directory {path}: {e}")

def copy_and_rename_files(base_source_dir, base_dest_dir, done_file):
    file_mapping = {
        "Data/Homology/images/Hom_Persistence_Diagrams_4_years.png": "Figure_3.2.png",
        "Data/Homology/images/Hom_PDs_Entropy_MIDs_and_FPCA_over_Years.png": "Figure_3.3.png",
        "Results/DGL_GNN_classification/plots/train_test_class_probabilities_proportions_over_years.png": "Figure_5.2.png",
        "Results/DGL_GNN_classification/plots/max_weights_entropy_mids.png": "Figure_5.3.png",
        "Results/DGL_GNN_classification/plots/subgraphs_top_8_edges_1960_1970.png": "Figure_5.4.png",
        "Results/ARGVA_results/latentspace/pca_kde/ARGVA_latent_space_PCA_features_kde_year_1980.png": "Figure_5.6.png",
        "Results/ARGVA_results/classifications/grouped_smoothed_individual_cohesion_cluster_analysis_test_data.png": "Figure_5.7.png",
        "Results/ARGVA_results/multiple_states_cluster_probabilities_over_years.png": "Figure_5.8.png",
        "Results/SubGraphX_results/SubGraphx/SubgraphX_4_years.png": "Figure_5.9.png",
        "Results/Bayesian_results/MCMC_SubGraphX/Empirical_data_roc_curve_plot_testing_data.png": "Figure_5.10.png",
        "Results/Bayesian_results/MCMC_sociomatrix/vars_summary.tex": "Table_2.1.tex",
        "Results/Bayesian_results/MCMC_sociomatrix_homology/vars_summary.tex": "Table_4.1.tex",
        "Results/Bayesian_results/MCMC_sociomatrix_homology_latent/vars_summary.tex": "Table_5.1.tex",
        "Results/Bayesian_results/MCMC_SubGraphX/vars_summary.tex": "Table_5.2.tex",
        "Results/Bayesian_results/MCMC_sociomatrix/vars_summary.csv": "Table_2.1.csv",
        "Results/Bayesian_results/MCMC_sociomatrix_homology/vars_summary.csv": "Table_4.1.csv",
        "Results/Bayesian_results/MCMC_sociomatrix_homology_latent/vars_summary.csv": "Table_5.1.csv",
        "Results/Bayesian_results/MCMC_SubGraphX/vars_summary.csv": "Table_5.2.csv",
    }

    create_directory(base_dest_dir)

    for source_rel, new_name in file_mapping.items():
        src = os.path.join(base_source_dir, source_rel)
        dst = os.path.join(base_dest_dir, new_name)

        try:
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"[✔] Copied: {new_name}")
            else:
                print(f"[⚠] Missing: {source_rel}")
        except Exception as e:
            print(f"[❌] Error copying {source_rel}: {e}")

    Path(done_file).parent.mkdir(parents=True, exist_ok=True)
    Path(done_file).touch()

def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    base_source = base_dir / "project_data"
    base_dest = base_dir / "paper_figures_and_tables_extracted_from_results_folder/"
    done_marker = base_dir / "logs" / "8_extract_paper_figures_and_tables_to_new_folder.py.log"


    print("📁 Extracting and renaming results...")
    copy_and_rename_files(base_source, base_dest, done_marker)
    print("✅ Extraction completed.")

if __name__ == "__main__":
    main()


