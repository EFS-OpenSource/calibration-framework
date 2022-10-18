# Visualization Package

Methods for the visualization of miscalibration in the scope of confidence calibration and regression uncertainty
calibration.

This package consists of a *netcal.presentation.ReliabilityDiagram* (Reliability Diagram) method used to
visualize the calibration properties for confidence
calibration in the scope of classification, object detection (semantic label confidence) or segmentation.
Similar to the ACE or ECE, this method bins the samples in equally sized bins by their confidence and
displays the gap between confidence and observed accuracy in each bin.

For regression uncertainty calibration, this package also holds the *netcal.presentation.ReliabilityRegression*
method that is able to visualize the quantile calibration properties of probabilistic regression models, e.g., within
probabilistic regression or object detection (spatial position uncertainty).
A complementary diagram is the *netcal.presentation.ReliabilityQCE* that visualizes the computation of the *netcal.metrics.regression.QCE* (C-QCE) metric.