# Directory Structure for `project_data`

```
project_data
├── **Data/**
│   ├── **Data_original/**
│   ├── **Data_prepared/**
│   │   ├── **dyads_DV&IDVs_pol_igo/**
│   │   ├── **dyads_IOmembs_pol_igo/**
│   │   ├── **states_DV&IDVs_pol_igo/**
│   │   ├── **states_IOmembs_pol_igo/**
│   │   └── **states_pol_igos/**
│   └── **Homology/**
│       ├── **Homology_categorized/**
│       │   ├── **H_1_1/**
│       │   ├── **H_1_2/**
│       │   ├── **H_2_1/**
│       │   └── **H_2_2/**
│       ├── **IO_data_hom/**
│       ├── **Images/**
│       ├── **Persistence_Diagrams/**
│       ├── **Persistence_Images/**
│       └── **Persistence_Landscapes/**
└── **Results/**
    ├── **ARGVA_results/**
    │   └── **epochs/**
    │       ├── **classifications/**
    │       ├── **generated_graphs/**
    │       ├── **latentspace/**
    │       │   ├── **latentspace_csv/**
    │       │   └── **pca_kde/**
    │       ├── **losses/**
    │       │   └── **plots/**
    │       └── **saved_ARGVA_models/**
    ├── **Bayesian_results/**
    │ 
    ├── **DGL_GNN_Classification/**
    │   └── **nepochs/**
    │       ├── **GNN_models/**
    │       │   └── **version/**
    │       └── **plots/**
    └── **SubGraphX_results/**
        └── **nepochs/**
            ├── **Reconstructed_data/**
            ├── **SubGraphX/**
            └── **subgraphx_model_logs/**
                └── **version/**
```
