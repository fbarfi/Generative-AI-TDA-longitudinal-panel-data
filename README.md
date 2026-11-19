# Generative AI and Topological Data Analysis of Longitudinal Panel Data

Replication code for

> Arfi, Badredine. 2026. "Generative AI and Topological Data Analysis of Longitudinal Panel Data." *Political Analysis* 34 (1): 109–131. https://doi.org/10.1017/pan.2025.10019

The raw input data are not tracked in git; they are attached to the release `data-v1` and restored by one command (see [Data](#data)).

The pipeline applies topological data analysis (GUDHI persistence diagrams), graph neural network classification (DGL, with attention-weight analysis), an adversarially regularized variational graph autoencoder (ARGVA), Bayesian models, and SubGraphX explanations to dyadic militarized interstate dispute (MID) panel data, 1946–2010.

## Repository layout

```
.
├── conda/                         Pipeline run in eight Conda environments
│   ├── project/
│   │   ├── scripts_master/        Stage drivers 1–8 (run in order)
│   │   ├── project_codes/         Stage code: <stage>_main_codes/{main_*.py, <stage>_python_codes/}
│   │   └── project_data/
│   │       ├── Data/              Data_original (restored by data/fetch_data.py), Data_prepared, Homology
│   │       └── Results/           ARGVA, Bayesian, DGL_GNN_Classification, SubGraphX outputs
│   └── utilities_scripts/
│       ├── conda_env_ymls/        env_1 … env_7 (exported environments)
│       └── create_envs_from_yml.py
├── docker/                        Same pipeline run in Docker images
│   ├── docker_building_scripts/   Dockerfile<stage> + build_dockers_<stage>.py
│   └── docker_execution/          scripts_master/, project_codes/, project_data/, run_all.sh
├── data/
│   ├── fetch_data.py              Restores Data_original from the data-v1 release and verifies checksums
│   ├── make_data_release.py       Builds the data-v1 zips (author's tool)
│   ├── conda_Data_original.sha256
│   └── docker_Data_original.sha256
├── docs/                          Author guidance (overview PDF, Conda guide, Docker build guide, directory maps)
└── paper_figures/                 Figures as published in the article
```

Directory maps of `project_codes` and `project_data` are in `docs/project_codes_structure.md` and `docs/project_data_structure.md`. Empty output directories are kept with `.gitkeep` files because parts of the stage code write into them without creating them.

## Data

All inputs are publicly available third-party datasets (Correlates of War MID, IGO, alliance, contiguity, NMC, state system and major-power data; Polity5; Gleditsch capital-distance, minimum-distance and GDP data). Sources and citations are listed in `docs/readme_overview.pdf`, Table 1.

`Data_original` holds 31 files, about 670 MB per pipeline version. `dyadic_format_v3.dta` (≈518 MB) exceeds GitHub's 100 MB file limit and `dyadic_trade_COW_4.0.csv` (≈82 MB) exceeds the 50 MB warning threshold. The files are therefore attached as two zip files to the release [`data-v1`](https://github.com/fbarfi/Generative-AI-TDA-longitudinal-panel-data/releases/tag/data-v1) of this repository, `Data_original_conda.zip` and `Data_original_docker.zip`, and restored with

```bash
python data/fetch_data.py                       # download both zips from the release and restore both trees
python data/fetch_data.py --only conda          # restore one tree only
python data/fetch_data.py --zip-dir ~/Downloads # use zips downloaded by hand from the release page
```

Each restored file is verified against the SHA-256 manifests in `data/`. The script uses the Python standard library only.

## Running the pipeline

Stages must run in order; each stage reads intermediate outputs of earlier stages. Each driver in `scripts_master/` writes `logs/`, `done_flags/` and `execution_results_csv/` next to itself and marks completed sub-scripts in `done_flags/`. Drivers 1, 2.2 and 4–7 skip sub-scripts already marked; drivers 2.1 and 3 rerun them.

| Stage | Driver | Conda env | Content |
|---|---|---|---|
| 1 | `1_run_all_data_scripts.py` | `env_1` | Data preparation, dyadic sociomatrices |
| 2.1 | `2_1_run_all_gudhi_scripts.py` | `env_2_1` | Persistent homology (GUDHI) |
| 2.2 | `2_2_run_graph_data_scripts.py` | `env_2_2` | Graph construction |
| 3 | `3_run_all_dgl_classification_scripts.py` | `env_3` | DGL GNN classification, clustering, attention analysis |
| 4 | `4_run_all_argva_scripts.py` | `env_4` | ARGVA generative model |
| 5 | `5_run_all_bayesian_scripts.py` | `env_5` | Bayesian models |
| 6 | `6_run_all_subgraphx_scripts.py` | `env_6` | SubGraphX (longest stage) |
| 7 | `7_run_all_bayesian_subgraphx_scripts.py` | `env_7` | Bayesian model on SubGraphX output |
| 8 | `8_extract_paper_figures_and_tables_to_new_folder.py` | any | Copies paper figures and tables to `paper_figures_and_tables_extracted_from_results_folder/` |

### Conda

```bash
python data/fetch_data.py --only conda
cd conda/utilities_scripts && python create_envs_from_yml.py && cd -
cd conda/project/scripts_master
conda activate env_1 && python 1_run_all_data_scripts.py && conda deactivate
# … repeat for stages 2.1–7 with the matching environment …
python 8_extract_paper_figures_and_tables_to_new_folder.py
```

Full instructions: `docs/running_scripts_in_conda_env_guide.md`. The environment files were exported on Apple Silicon macOS and pin `osx-arm64` build strings; on other platforms the build strings may need to be relaxed.

### Docker

```bash
python data/fetch_data.py --only docker
cd docker/docker_building_scripts && python build_dockers_<stage>.py   # one per stage
cd ../docker_execution && bash run_all.sh
```

`run_all.sh` mounts `project_data/`, `project_codes/`, `scripts_master/` and `logs/` into each image and runs the stages in order. Build instructions: `docs/docker_build_guide.md`.

## Differences between the Conda and Docker versions

The two trees are archived as separate versions and are reproduced here unchanged. Their code differs in the following settings.

| File | Setting | `conda/` | `docker/` |
|---|---|---|---|
| `DGL_classification_main_codes/main_train_and_plot.py` | `nepochs` | 1000 | 500 |
| | `gradient_clip_val` | 1.0 | not set |
| | device / accelerator | commented out | CPU |
| `DGL_classification_main_codes/main_clustering.py`, `main_attention_analysis.py` | `nepochs` | 2000 | 1000 |
| | `weight_decay`, `patience` | 1e-5, 5 | 1e-4, 20 |
| | `val_ratio` | 0.4 | 0.2 |
| | device / accelerator | commented out | CPU |
| `DGL_GNN_classification_python_codes/gnntrainer.py` | `accelerator` argument | commented out | passed to Trainer |
| `DGL_GNN_classification_python_codes/graphclusteranalysis.py` | cluster assignment | NumPy k-means on saved per-phase outputs | scikit-learn k-means on reloaded model embeddings |
| `DGL_GNN_classification_python_codes/attentionweightsanalysis.py` | years plotted | 1960, 1970 | 1985, 2010 |
| `ARGVA_main_codes/main_ARVGA.py` | `batch_size` | 10 | 5 |
| `SubGraphX_main_codes/main_SubGraphX.py` | years | 1946–2010 | 1985–2010 |
| | `num_child`, `node_min`, `shapley_steps` | 5, 5, 5 | 12, 15, 80 |
| `bayesian_main_codes/main_bayesian.py` | `n_iter` | 5000 | 2000 |
| `scripts_master/8_extract_…py` | Figure 5.4 source | `subgraphs_top_8_edges_1960_1970.png` | `subgraphs_top_8_edges_1985_2010.png` |

`Data_original/data_lacp.csv` also differs between the two archived trees (see the SHA-256 manifests). The file is an output of `processdatafromsources.py` and is regenerated in stage 1.

## Figures and tables

Stage 8 maps result files to the article's figures and tables (Figures 3.2, 3.3, 5.2–5.4, 5.6–5.10; Tables 2.1, 4.1, 5.1, 5.2). The mapping, with the producing code for each item, is listed in `docs/readme_overview.pdf`. The figures as published are in `paper_figures/`.

## Computational notes

The author's runs used an Apple Silicon M4 machine (16 cores, 128 GB unified memory). A GPU is not required. PyTorch Lightning allows CUDA use; the graph libraries do not support the Apple `mps` backend. A full run takes many hours; stage 6 is the longest.

## Citation

Cite the article. Machine-readable metadata are in `CITATION.cff`.

```bibtex
@article{arfi2026generative,
  author  = {Arfi, Badredine},
  title   = {Generative {AI} and Topological Data Analysis of Longitudinal Panel Data},
  journal = {Political Analysis},
  year    = {2026},
  volume  = {34},
  number  = {1},
  pages   = {109--131},
  doi     = {10.1017/pan.2025.10019}
}
```

## License

CC0 1.0 Universal (public domain dedication). See `LICENSE`.

The CC0 dedication covers the code and documentation in this repository. The third-party input datasets restored by `data/fetch_data.py` remain subject to the terms of their original providers.

## Author

Badredine Arfi, Department of Political Science, University of Florida — barfi@ufl.edu
