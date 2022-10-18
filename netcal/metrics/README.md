# Metrics Package to Measure Miscalibration

Methods for measuring miscalibration in the context of confidence calibration and regression uncertainty calibration.

The common methods for confidence calibration evaluation are given with the
*netcal.metrics.confidence.ECE* (ECE), *netcal.metrics.confidence.MCE* (MCE), and
*netcal.metrics.confidence.ACE* (ACE). Each method bins the samples by their confidence and measures the
accuracy in each bin. The ECE gives the mean gap between confidence and observed accuracy in each bin weighted by the
number of samples. The MCE returns the highest observed deviation. The ACE is similar to the ECE but weights
each bin equally.

The common methods for regression uncertainty evaluation are *netcal.metrics.regression.PinballLoss* (Pinball
loss), the *netcal.metrics.regression.NLL* (NLL), and the *netcal.metrics.regression.QCE* (M-QCE and
C-QCE). The Pinball loss as well as the Marginal/Conditional Quantile Calibration Error (M-QCE and C-QCE) evaluate
the quality of the estimated quantiles compared to the observed ground-truth quantile coverage. The NLL is a proper
scoring rule to measure the overall quality of the predicted probability distributions.

For a detailed description of the available metrics within regression calibration, see the module doc of
*netcal.regression*.