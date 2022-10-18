# Regression GP Calibration Package

This package provides the framework for all Gaussian Process (GP) recalibration schemes.
These are *GP-Beta* [2], GP-Normal [3], and GP-Cauchy [3]. The goal of regression calibration using a GP scheme is to achieve
*distribution calibration*, i.e., to match the predicted moments (mean, variance) to the true observed ones.
In contrast to *quantile calibration* [1], where only the marginal calibration is of interest, the *distribution calibration* [2] is more
restrictive. It requires that the predicted moments should match the observed ones *given a certain probability
distribution*. Therefore, the authors in [2] propose to use Gaussian process to estimate the recalibration
parameters of a Beta calibration function locally (i.e., matching the observed moments of neighboring samples).
The *GP-Normal* and the *GP-Cauchy* follow the same principle but return parametric output distributions after calibration.


## References

[1] Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon:
   "Accurate uncertainties for deep learning using calibrated regression."
   International Conference on Machine Learning. PMLR, 2018.
   [Get source online](http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf)

[2] Hao Song, Tom Diethe, Meelis Kull and Peter Flach:
   "Distribution calibration for regression."
   International Conference on Machine Learning. PMLR, 2019.
   [Get source online](http://proceedings.mlr.press/v97/song19a/song19a.pdf)

[3] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
   "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
   ArXiv preprint arXiv:2207.01242, 2022.
   [Get source online](https://arxiv.org/pdf/2207.01242.pdf)