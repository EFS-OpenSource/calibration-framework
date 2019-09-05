Calibration Framework
=====================
Calibration framework in Python 3 for Neural Networks.

Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
AND Visteon Electronics Germany GmbH, Kerpen, Germany

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

.. contents:: Table of Contents
   :depth: 2

Overview
===============

This framework is designed to calibrate the confidence estimates of classifiers like Neural Networks. Modern Neural Networks are likely to be overconfident with their predictions. However, reliable confidence estimates of such classifiers are crucial especially in safety-critical applications.

For example: given 100 predictions with a confidence of 80% of each prediction, the observed accuracy should also match 80% (neither more nor less). This behaviour is achievable with several calibration methods.

The framework is structured as follows::

    netcal
      .binning         # binning methods
      .scaling         # scaling methods
      .regularization  # regularization methods
      .presentation    # presentation methods
      .metrics         # metrics for measuring miscalibration

    examples           # example code snippets

Installation
===============
The installation of the calibration suite is quite easy with setuptools. You can either install this framework using PIP::

    pip3 install netcal

Or simply invoke the following command to install the calibration suite::

    python3 setup.py install

Calibration Metrics
======================
The most common metric to determine miscalibration is the *Expected Calibration Error* (ECE) [1]_. This metric divides the confidence space into several bins and measures the observed accuracy in each bin. The bin gaps between observed accuracy and bin confidence are summed up and weighted by the amount of samples in each bin. The *Maximum Calibration Error* (MCE) denotes the highest gap over all bins. The *Average Calibration Error* (ACE) [11]_ denotes the average miscalibration where each bin gets weighted equally.

Another group are the regularization tools which are added to the loss during the training of a Neural Network.

Methods
==========
The calibration methods are separated into binning and scaling methods. The binning methods divide the confidence space into several bins (like ECE) and perform calibration on each bin. The scaling methods scale the confidence estimates or logits directly to calibrated confidence estimates.

Most of the calibration methods are designed for binary classification tasks. Multi-class calibration is performed in "one vs. all" by default.

Some methods like "Isotonic Regression" utilize methods from the scikit-learn API [9]_.

Binning
-------
Implemented binning methods are:

- Histogram Binning [3]_, [4]_
- Isotonic Regression [4]_, [5]_
- Bayesian Binning into Quantiles (BBQ) [1]_
- Ensemble of Near Isotonic Regression (ENIR) [6]_

Scaling
-------
Implemented scaling methods are:

- Logistic Calibration/Platt Scaling [10]_
- Temperature Scaling [7]_
- Beta Calibration [2]_

Regularization
--------------
Implemented regularization methods are:

- Confidence Penalty [8]_

Visualization
================
For visualization of miscalibration, one can use a Confidence Histograms & Reliability Diagrams. These diagrams are similar to ECE, the output space is divided into equally spaced bins. The calibration gap between bin accuracy and bin confidence is visualized as a histogram.

Examples
===========
The calibration methods work with the predicted confidence estimates of a Neural Network. This is a basic example which uses softmax predictions of a classification task with 10 classes and the given NumPy arrays::

    ground_truth  # this is a NumPy 1-D array with ground truth digits between 0-9 - shape: (n_samples,)
    confidences   # this is a NumPy 2-D array with confidence estimates between 0-1 - shape: (n_samples, n_classes)

This is an example for Temperature Scaling but also works for every calibration method (remind different constructor parameters)::

    import numpy as np
    from netcal.scaling import TemperatureScaling

    temperature = TemperatureScaling()
    temperature.fit(confidences, ground_truth)
    calibrated = temperature.transform(confidences)

The miscalibration can be determined with the ECE::

    from netcal.metrics import ECE

    n_bins = 10

    ece = ECE(n_bins)
    uncalibrated_score = ece.measure(confidences)
    calibrated_score = ece.measure(calibrated)

The miscalibration can be visualized with a Reliability Diagram::

    from netcal.presentation import ReliabilityDiagram

    n_bins = 10

    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(confidences, ground_truth)  # visualize miscalibration of uncalibrated
    diagram.plot(calibrated, ground_truth)   # visualize miscalibration of calibrated

References
==========
.. [1] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht: "Obtaining well calibrated probabilities using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
.. [2] Kull, Meelis, Telmo Silva Filho, and Peter Flach: "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers." Artificial Intelligence and Statistics, PMLR 54:623-631, 2017.
.. [3] Zadrozny, Bianca and Elkan, Charles: "Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers." In ICML, pp. 609–616, 2001.
.. [4] Zadrozny, Bianca and Elkan, Charles: "Transforming classifier scores into accurate multiclass probability estimates." In KDD, pp. 694–699, 2002.
.. [5] Ryan J Tibshirani, Holger Hoefling, and Robert Tibshirani: "Nearly-isotonic regression." Technometrics, 53(1):54–61, 2011.
.. [6] Naeini, Mahdi Pakdaman, and Gregory F. Cooper: "Binary classifier calibration using an ensemble of near isotonic regression models." 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.
.. [7] Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger: "On Calibration of Modern Neural Networks." Proceedings of the 34th International Conference on Machine Learning, 2017.
.. [8] Pereyra, G., Tucker, G., Chorowski, J., Kaiser, L. and Hinton, G.: “Regularizing neural networks by penalizing confident output distributions.” CoRR, 2017.
.. [9] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E.: "Scikit-learn: Machine Learning in Python." In Journal of Machine Learning Research, volume 12 pp 2825-2830, 2011.
.. [10] Platt, John: "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." Advances in large margin classifiers, 10(3): 61–74, 1999.
.. [11] Neumann, Lukas, Andrew Zisserman, and Andrea Vedaldi: "Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection." Conference on Neural Information Processing Systems (NIPS) Workshop MLITS, 2018.