# fastcryo
to make cryoem faster

- Project structure

```
.
├── README.md                 # The top-level description of content
├── .gitignore                # Specifies intentionally untracked files to ignore
├── requirements.txt          # Necessary packages for this project
│
├── src/                      # Source code for use in this project.
│   ├── __init__.py           # Makes src a Python module
│   ├── datasets/             # Scripts to process data
│   │   └── __init__.py
│   ├── models/               # Scripts to define models
│   │   ├── __init__.py
│   │   ├── model_1.py
│   │   └── ...
│   ├── utils/                # Utility functions and classes
│   │   └── __init__.py
│   ├── experiments/          # Scripts to run experiments
│   │   ├── experiment_1.py
│   │   └── ...
│   └── main.py               # Main script to run
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_model_1.py
│   └── ...
│
├── data/                     # Folder that contains data (ignore this folder in .gitignore)
│   ├── raw/                  # Raw data, immutable
│   ├── processed/            # Cleaned data, used for modelling
│   └── external/             # Data from third party sources
│
├── notebooks/                # Jupyter notebooks for analysis and prototyping
│   ├── EDA.ipynb
│   ├── experiment_1_results.ipynb
│   └── ...
│
├── results/                  # Folder for storing outputs of experiments and scripts
│   ├── experiment_1/         # Subfolder for each experiment
│   │   ├── metrics.json
│   │   ├── model.pt
│   │   └── ...
│   ├── experiment_2/
│   │   ├── metrics.json
│   │   ├── model.pt
│   │   └── ...
│   └── ...
│
└── docs/                     # Documentation for the project
    ├── report.pdf
    └── ...
```



# Create Dataset 
## with notebook
    - run 2023-0511_KatoLabDataParse.ipynb to create tiff files for only first 20frames
    - run 2023-0515_run_motionCor2.ipynb to use motionCor2
      

 - if you are interested in how MRC looks like and normal summation of TIFF, then look at these two
    - 2023-0517_KatoLabDataSum.ipynb
    - 2023-0517_MRC_vis.ipynb


- you might need to run this for your cuda to run motionCor2
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kyohei/miniconda3/envs/cryoem/lib



# Learn relion or cryoSPARC



## relion
- https://relion.readthedocs.io/en/release-4.0/Installation.html
- https://github.com/xtreme-d/relion-tutorial-simplified



# CryoEM General
- https://cryoem101.org/chapter-1/


# Kernel = "cryoem"


# Data in EMPIAR-11119

## EMPIAR


- https://www.ebi.ac.uk/empiar/EMPIAR-11119/

## FTP structure

- https://ftp.ebi.ac.uk/empiar/world_availability/11119/ 


- /empiar/world_availability/11119
    - 11119.xml
    - /data
        - CountRef_gpcr_190808no17_99-1_0000_Aug16_15.53.48.mrc
        - data/
            - gpcr_190808no17_0000_Aug17_20.59.59.tif
	    - gpcr_190808no17_99-1_0000_Aug16_15.53.48.tif
	    - gpcr_190808no17_99-1_0000_Aug16_15.53.48.tif.jpg
	    - more available in a metadata file (`filenames_ftp_data.parquet`)
