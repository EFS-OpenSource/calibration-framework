# Probabilistic Regression Calibration Package

Methods for uncertainty calibration of probabilistic regression tasks.
A probabilistic regression model does not only provide a continuous estimate but also an according uncertainty
(commonly a Gaussian standard deviation/variance).
The methods within this package are able to recalibrate this uncertainty by means of *quantile calibration* [1],
*distribution calibration* [2], or *variance calibration* [3], [4].

*Quantile calibration* [1] requires that a predicted quantile for a quantile level *t* covers
approx. *100t%* of the ground-truth samples.

Methods for *quantile calibration*:

- *IsotonicRegression* [1].

*Distribution calibration* [2] requires that a predicted probability distribution should be equal to the observed
error distribution. This must hold for all statistical moments.

Methods for *distribution calibration*:

- *GPBeta* [2].
- *GPNormal* [5].
- *GPCauchy* [5].

*Variance calibration* [3], [4] requires that the predicted variance of a Gaussian distribution should match the
observed error variance which is equivalent to the root mean squared error.

Methods for *variance calibration*:

- *VarianceScaling* [3], [4].
- *GPNormal* [5].

## References

[1] Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon:
   "Accurate uncertainties for deep learning using calibrated regression."
   International Conference on Machine Learning. PMLR, 2018.
   [Get source online](http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf)

[2] Hao Song, Tom Diethe, Meelis Kull and Peter Flach:
   "Distribution calibration for regression."
   International Conference on Machine Learning. PMLR, 2019.
   [Get source online](http://proceedings.mlr.press/v97/song19a/song19a.pdf)

[3] Levi, Dan, et al.:
   "Evaluating and calibrating uncertainty prediction in regression tasks."
   arXiv preprint arXiv:1905.11659 (2019).
   [Get source online](https://arxiv.org/pdf/1905.11659.pdf)

[4] Laves, Max-Heinrich, et al.:
   "Well-calibrated regression uncertainty in medical imaging with deep learning."
   Medical Imaging with Deep Learning. PMLR, 2020.
   [Get source online](http://proceedings.mlr.press/v121/laves20a/laves20a.pdf)

[5] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
   "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
   ArXiv preprint arXiv:2207.01242, 2022.
   [Get source online](https://arxiv.org/pdf/2207.01242.pdf)