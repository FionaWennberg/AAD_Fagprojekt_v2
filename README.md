````markdown
# aad_project

Project on AAD and the difference between normal hearing and hearing impaired users

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── scripts/                     # Data directory
│   └── run_all_subjects.py
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
    └──  statistics.ipynb
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── aad_project/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── preprocess_backward_eelbrain.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````
The GLMM statistics script uses pymer4, which calls R's lme4/lmerTest through rpy2.
Therefore, R must be installed separately and visible from Python.

Setup:
1. Install Python packages:
   pip install -r requirements.txt

2. Install R packages:
   Rscript requirements_R.R

3. On Windows, if rpy2 cannot find R, set:
   set RPY2_CFFI_MODE=ABI
   set R_HOME=C:\Program Files\R\R-4.4.3
   set PATH=%R_HOME%\bin\x64;%PATH%