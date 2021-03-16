Calibration Framework
=====================
Calibration framework in Python 3 for Neural Networks.
For full API reference documentation, visit https://fabiankueppers.github.io/calibration-framework.

Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
AND Visteon Electronics Germany GmbH, Kerpen, Germany

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

If you use this framework or parts of it for your research, please cite it by::

    @InProceedings{Kueppers_2020_CVPR_Workshops,
       author = {Küppers, Fabian and Kronenberger, Jan and Shantia, Amirhossein and Haselhoff, Anselm},
       title = {Multivariate Confidence Calibration for Object Detection},
       booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
       month = {June},
       year = {2020}
    }

.. contents:: Table of Contents
   :depth: 2

Overview
===============

This framework is designed to calibrate the confidence estimates of classifiers like neural networks. Modern neural networks are likely to be overconfident with their predictions. However, reliable confidence estimates of such classifiers are crucial especially in safety-critical applications.

For example: given 100 predictions with a confidence of 80% of each prediction, the observed accuracy should also match 80% (neither more nor less). This behaviour is achievable with several calibration methods.

This framework can also be used to calibrate object detection models. It has recently been shown that calibration on object detection also depends on the position and/or scale of a predicted object [12]_. We provide calibration methods to perform confidence calibration w.r.t. the additional box regression branch.
For this purpose, we extended the commonly used Histogram Binning [3]_, Logistic Calibration alias Platt scaling [10]_ and the Beta Calibration method [2]_ to also include the bounding box information into a calibration mapping.
Furthermore, we provide two new methods called the *Dependent Logistic Calibration* and the *Dependent Beta Calibration* that are not only able to perform a calibration mapping
w.r.t. additional bounding box information but also to model correlations and dependencies between all given quantities [12]_. Those methods should be preffered over their counterparts in object detection mode.

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

Requirements
------------
- numpy>=1.15
- scipy>=1.3
- matplotlib>=3.1
- scikit-learn>=0.20.0
- torch>=1.1
- tqdm


Calibration Metrics
======================
The most common metric to determine miscalibration in the scope of classification is the *Expected Calibration Error* (ECE) [1]_. This metric divides the confidence space into several bins and measures the observed accuracy in each bin. The bin gaps between observed accuracy and bin confidence are summed up and weighted by the amount of samples in each bin. The *Maximum Calibration Error* (MCE) denotes the highest gap over all bins. The *Average Calibration Error* (ACE) [11]_ denotes the average miscalibration where each bin gets weighted equally.
For object detection, we implemented the *Detection Calibration Error* (D-ECE) [12]_ that is the natural extension of the ECE to object detection tasks. The miscalibration is determined w.r.t. the bounding box information provided (e.g. box location and/or scale). For this purpose, all available information gets binned in a multidimensional histogram. The accuracy is then calculated in each bin separately to determine the mean deviation between confidence and accuracy.

- (Detection) Expected Calibration Error [1]_, [12]_ (*netcal.metrics.ECE*)
- (Detection) Maximum Calibration Error [1]_, [12]_  (*netcal.metrics.MCE*)
- (Detection) Average Calibration Error [11]_, [12]_ (*netcal.metrics.ACE*)

Methods
==========
The calibration methods are separated into binning and scaling methods. The binning methods divide the available information into several bins (like ECE or D-ECE) and perform calibration on each bin. The scaling methods scale the confidence estimates or logits directly to calibrated confidence estimates - on detection calibration, this is done w.r.t. the additional regression branch of a network.

Important: if you use the detection mode, you need to specifiy the flag "detection=True" in the constructor of the according method (this is not necessary for *netcal.scaling.LogisticCalibrationDependent* and *netcal.scaling.BetaCalibrationDependent*).

Most of the calibration methods are designed for binary classification tasks. For binning methods, multi-class calibration is performed in "one vs. all" by default.

Some methods like "Isotonic Regression" utilize methods from the scikit-learn API [9]_.

Another group are the regularization tools which are added to the loss during the training of a Neural Network.

Binning
-------
Implemented binning methods are:

- Histogram Binning for classification [3]_, [4]_ and object detection [12]_ (*netcal.binning.HistogramBinning*)
- Isotonic Regression [4]_, [5]_ (*netcal.binning.IsotonicRegression*)
- Bayesian Binning into Quantiles (BBQ) [1]_ (*netcal.binning.BBQ*)
- Ensemble of Near Isotonic Regression (ENIR) [6]_ (*netcal.binning.ENIR*)

Scaling
-------
Implemented scaling methods are:

- Logistic Calibration/Platt Scaling for classification [10]_, [12]_ and object detection [12]_ (*netcal.scaling.LogisticCalibration*)
- Dependent Logistic Calibration for object detection [12]_ (*netcal.scaling.LogisticCalibrationDependent*) - on detection, this method is able to capture correlations between all input quantities and should be preferred over Logistic Calibration for object detection
- Temperature Scaling for classification [7]_ and object detection [12]_ (*netcal.scaling.TemperatureScaling*)
- Beta Calibration for classification [2]_ and object detection [12]_ (*netcal.scaling.BetaCalibration*)
- Dependent Beta Calibration for object detection [12]_ (*netcal.scaling.BetaCalibrationDependent*) - on detection, this method is able to capture correlations between all input quantities and should be preferred over Beta Calibration for object detection

Regularization
--------------
Implemented regularization methods are:

- Confidence Penalty [8]_ (*netcal.regularization.confidence_penalty*)

Visualization
================
For visualization of miscalibration, one can use a Confidence Histograms & Reliability Diagrams. These diagrams are similar to ECE, the output space is divided into equally spaced bins. The calibration gap between bin accuracy and bin confidence is visualized as a histogram.

On detection calibration, the miscalibration can be visualized either along one additional box information (e.g. the x-position of the predictions) or distributed over two additional box information in terms of a heatmap.

- Reliability Diagram [1]_, [12]_ (*netcal.presentation.ReliabilityDiagram*)

Examples
===========
The calibration methods work with the predicted confidence estimates of a neural network and on detection also with the bounding box regression branch.

Classification
--------------
This is a basic example which uses softmax predictions of a classification task with 10 classes and the given NumPy arrays:

.. code-block:: python

    ground_truth  # this is a NumPy 1-D array with ground truth digits between 0-9 - shape: (n_samples,)
    confidences   # this is a NumPy 2-D array with confidence estimates between 0-1 - shape: (n_samples, n_classes)

This is an example for *netcal.scaling.TemperatureScaling* but also works for every calibration method (remind different constructor parameters):

.. code-block:: python

    import numpy as np
    from netcal.scaling import TemperatureScaling

    temperature = TemperatureScaling()
    temperature.fit(confidences, ground_truth)
    calibrated = temperature.transform(confidences)

The miscalibration can be determined with the ECE:

.. code-block:: python

    from netcal.metrics import ECE

    n_bins = 10

    ece = ECE(n_bins)
    uncalibrated_score = ece.measure(confidences)
    calibrated_score = ece.measure(calibrated)

The miscalibration can be visualized with a Reliability Diagram:

.. code-block:: python

    from netcal.presentation import ReliabilityDiagram

    n_bins = 10

    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(confidences, ground_truth)  # visualize miscalibration of uncalibrated
    diagram.plot(calibrated, ground_truth)   # visualize miscalibration of calibrated

Detection
---------
This is a basic example which uses softmax predictions of a classification task with 10 classes and the given NumPy arrays:

.. code-block:: python

    matched                # binary NumPy 1-D array (0, 1) that indicates if a bounding box has matched a ground truth at a certain IoU with the right label - shape: (n_samples,)
    confidences            # NumPy 1-D array with confidence estimates between 0-1 - shape: (n_samples,)
    relative_x_position    # NumPy 1-D array with relative center-x position between 0-1 of each prediction - shape: (n_samples,)

This is an example for *netcal.scaling.LogisticCalibration* and *netcal.scaling.LogisticCalibrationDependent* but also works for every calibration method (remind different constructor parameters):

.. code-block:: python

    import numpy as np
    from netcal.scaling import LogisticCalibration, LogisticCalibrationDependent

    input = np.stack((confidences, relative_x_position), axis=1)

    lr = LogisticCalibration(detection=True)        # flag 'detection=True' is mandatory for this method
    lr.fit(input, matched)
    calibrated = lr.transform(input)

    lr_dependent = LogisticCalibrationDependent()   # flag 'detection=True' is not necessary as this method is only defined for detection
    lr_dependent.fit(input, matched)
    calibrated = lr_dependent.transform(input)

The miscalibration can be determined with the D-ECE:

.. code-block:: python

    from netcal.metrics import ECE

    n_bins = [10, 10]
    input_calibrated = np.stack((calibrated, relative_x_position), axis=1)

    ece = ECE(n_bins, detection=True)           # flag 'detection=True' is mandatory for this method
    uncalibrated_score = ece.measure(input, matched)
    calibrated_score = ece.measure(input_calibrated, matched)

The miscalibration can be visualized with a Reliability Diagram:

.. code-block:: python

    from netcal.presentation import ReliabilityDiagram

    n_bins = [10, 10]

    diagram = ReliabilityDiagram(n_bins, detection=True)    # flag 'detection=True' is mandatory for this method
    diagram.plot(input, matched)                # visualize miscalibration of uncalibrated
    diagram.plot(input_calibrated, matched)     # visualize miscalibration of calibrated

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
.. [12] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff: "Multivariate Confidence Calibration for Object Detection"." The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020
