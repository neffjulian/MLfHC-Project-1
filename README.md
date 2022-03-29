
# ML4H course project

This is the course project for Machine Learning for Health Care

## Installation

Create the conda environment for the project. A version of conda must be installed on your system.

```bash
  conda env create -f environment.yml
  conda activate ml4h
```

## Dataset

Download the dataset from moodle and extract it to the `data` folder

    .
    ├── ...
    ├── data                    
    │   ├── mitbih_test.csv
    │   ├── mitbih_train.csv
    │   ├── ptbdb_abnormal.csv
    │   ├── ptbdb_normal.csv
    └── ...

## Experiments

Each experiment has its own config file in the configs folder. An experiment can be run through

```bash
  python main.py --config configs/base/mitbih_baseline_cnn.yaml
```

## Authors

- [@Julian Neff](https://github.com/neffjulian)
- [@Michael Mazourik](https://github.com/MikeDoes)
- [@Remo Kellenberger](https://github.com/remo48)

## Appendix

Overleaf: https://www.overleaf.com/project/6230698472ef0731f2b54470
