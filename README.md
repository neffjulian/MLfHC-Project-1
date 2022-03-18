
# ML4H course project

This is the course project for Machine Learning for Health Care

## Installation

Create conda environment for project. A version of conda must be installed on your system

```bash
  conda env create -f environment.yml
  conda activate ml4h
```

### Update Environment

If there are changes in the environment file update them using

```bash
  conda env update -f environment.yml --prune
```

## Dataset

Download the dataset from moodle and extract to `data`

    .
    ├── ...
    ├── data                    
    │   ├── mitbih_test.csv
    │   ├── mitbih_train.csv
    │   ├── ptbdb_abnormal.csv
    │   ├── ptbdb_normal.csv
    └── ...


## Authors

- [@Julian Neff](https://github.com/neffjulian)
- [@Michael Mazourik](https://github.com/MikeDoes)
- [@Remo Kellenberger](https://github.com/remo48)

## Appendix

Overleaf: https://www.overleaf.com/project/6230698472ef0731f2b54470
