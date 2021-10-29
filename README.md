# Gravitational Wave Detection

Solution to the [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/seti-breakthrough-listen) Kaggle competition on classifying gravitational wave signals from black hole collisions.

![sample](https://i.postimg.cc/FzPd8TTJ/black-hole.jpg)


## Summary

Gravitational waves are tiny ripples in the fabric of space-time. Even though researchers have built a global network of gravitational wave detectors, including some of the most sensitive instruments on the planet, the obtained signals are buried in detector noise. Recovering the signals allows scientists to observe a new population of massive, stellar-origin black holes, to unlock the mysteries of neutron star mergers, and to measure the expansion of the Universe.

This project uses deep learning to detect gravitational wave time-series signals from the mergers of black holes. The modeling pipeline involves converting time series into 2D images and employing computer vision models. My solution is a blend of multiple EfficientNet CNNs. All models are implemented in `PyTorch`.


## Project structure

The project has the following structure:
- `codes/`: `.py` main scripts with data, model, training and inference modules
- `notebooks/`: `.ipynb` Colab-friendly notebooks with model training and blending
- `input/`: input data not included due to size limits and can be downloaded [here](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)
- `output/`: model configurations, predictions and figures exported from the notebooks


## Working with the repo

### Environment

To work with the repo, I recommend to create a virtual Conda environment from the `environment.yml` file:
```
conda env create --name g2net --file environment.yml
conda activate g2net
```

### Reproducing solution

The solution can be reproduced in the following steps:
1. Download input data and place it in the `input/` folder.
1. Run training notebooks `training_v1.ipynb` to `training_v9.ipynb` to obtain model predictions.
3. Run the ensembling notebook `ensembling.ipynb` to obtain the final predictions.

More details are provided in the documentation within the scripts & notebooks.
