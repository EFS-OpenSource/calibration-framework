# Metrics for Regression Uncertainty Calibration

Methods for measuring miscalibration in the context of regression uncertainty calibration for probabilistic
regression models.

The common methods for regression uncertainty evaluation are *netcal.metrics.regression.PinballLoss* (Pinball
loss), the *netcal.metrics.regression.NLL* (NLL), and the *netcal.metrics.regression.QCE* (M-QCE and
C-QCE). The Pinball loss as well as the Marginal/Conditional Quantile Calibration Error (M-QCE and C-QCE) evaluate
the quality of the estimated quantiles compared to the observed ground-truth quantile coverage. The NLL is a proper
scoring rule to measure the overall quality of the predicted probability distributions.

Further metrics are the *netcal.metrics.regression.UCE* (UCE) and the *netcal.metrics.regression.ENCE*
(ENCE) which both apply a binning scheme over the predicted standard deviation/variance and test for *variance
calibration*.

For a detailed description of the available metrics within regression calibration, see the module doc of
*netcal.regression*.