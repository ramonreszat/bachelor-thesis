# Graph-based Data Integration
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)

This is my bachelor thesis at [Maastricht University](https://www.maastrichtuniversity.nl/education/bachelor/data-science-and-artificial-intelligence) implementing machine learning methods on multi-omics data.

## Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/).
```console
user@home:~$ python3 -m venv thesis
user@home:~$ source thesis/bin/activate
user@home:~$ pip3 install -r requirements.txt
```

## Notebooks

1. [Dataset of Multi-omics Measurements from Cancer Cells](notebooks/multi-omics.ipynb)
2. [Dataset of Drug Sensitivity Assays for Anticancer Compounds](notebooks/drug-sensitivity.ipynb)
3. [Graph-based Data Integration Methods](notebooks/affinity-graphs.ipynb)
4. [Phenotypic Subtype Discovery by Multi-omics Clustering](notebooks/clustering.ipynb)
5. [Drug Sensitivity Screening from Clustering Results](notebooks/biomarkers.ipynb)

## Datasets

The following datasets are used in this project:
- [Cancer Cell Line Encyclopedia](https://sites.broadinstitute.org/ccle/)
- [Cancer Therapeutic Response Portal](https://portals.broadinstitute.org/ctrp.v2.1/)

## References

Also refer to the original article introducing [affinity network fusion](https://www.sciencedirect.com/science/article/pii/S1046202317304930) \[1\] and the corresponding [R](https://github.com/BeautyOfWeb/ANF) implementation.

    [1] Ma, T., & Zhang, A. (2018). Affinity network fusion and semi-supervised learning for cancer
    patient clustering. Methods, 145, 16-24.
    
    [2] Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Tu, Z., Brudno, M., Haibe-Kains, B., &
    Goldenberg, A. (2014). Similarity network fusion for aggregating data types on a genomic scale.
    Nature Methods, 11(3), 333.
