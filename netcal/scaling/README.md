# Scaling Methods for Confidence Calibration

This package consists of several methods for confidence calibration which use confidence scaling to approximate
confidence estimates to observed accuracy. The most common scaling methods are
*netcal.scaling.TemperatureScaling*, *netcal.scaling.LogisticCalibration*, and *netcal.scaling.BetaCalibration*.
Note that all methods can also be applied to object detection and are capable of additional influenting factors
such as object position and/or shape.
The advanced methods *netcal.scaling.LogisticCalibrationDependent* and *netcal.scaling.BetaCalibrationDependent*
are able to better represent possible correlations as the underlying probability distributions are joint
multivariate distributions with possible correlations.