# Metrics for Confidence Calibration

Methods for measuring miscalibration in the context of confidence calibration.

The common methods for confidence calibration evaluation are given with the
*netcal.metrics.confidence.ECE* (ECE), *netcal.metrics.confidence.MCE* (MCE), and
*netcal.metrics.confidence.ACE* (ACE). Each method bins the samples by their confidence and measures the
accuracy in each bin. The ECE gives the mean gap between confidence and observed accuracy in each bin weighted by the
number of samples. The MCE returns the highest observed deviation. The ACE is similar to the ECE but weights
each bin equally.

A further metric is the Maximum Mean Calibration Error (MMCE) which is a differentiable variant of the ECE that
might also be used as a regularization technique during model training.