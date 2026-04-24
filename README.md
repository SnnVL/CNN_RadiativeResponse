# Convolutional Neural Network (CNN) code for map to value predictions

Code used to train a CNN to predict global mean radiation ($R$) from surface temperature maps. The framework is set up to generally test different architectures for predicting a single value from 2D input. Postprocessing to apply the CNN to different datasets is done in jupyter notebooks. For example, `FeedbackManuscriptPlots.ipynb` contains code to generate the plots in [1].

Code based on templates from [victoresque](https://github.com/victoresque/pytorch-template) and [eabarnes1010](https://github.com/eabarnes1010/pytorch_template), including [shash_peak_warming](https://github.com/eabarnes1010/shash_peak_warming_public).

## Installation

```bash
conda create --name env_pt python=3.13
conda activate env_pt
conda install numpy scipy pandas matplotlib jupyterlab xarray scikit-learn cartopy netCDF4 pytorch
pip install cmcrameri cmocean cmasher cmaps torchinfo graphviz
conda install conda-forge xcdat
```

Set correct directories to use in `utils/DIRECTORIES.py`.

## References

[1] S. Van Loon, M. Rugenstein, Mark D. Zelinka, and Timothy Andrews, "Recent Weakening of the Global Radiative Feedback", [arXiv:2603.12515](https://doi.org/10.48550/arXiv.2603.12515) (2026)
