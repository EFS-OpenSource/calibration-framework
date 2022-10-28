# net:cal - Uncertainty Calibration

<div style="text-align: justify">

The **net:cal** calibration framework is a Python 3 library for measuring and mitigating miscalibration of uncertainty estimates, e.g., by a neural network. 
For full API reference documentation, visit
<https://efs-opensource.github.io/calibration-framework>.

Copyright &copy; 2019-2022 Ruhr West University of Applied Sciences,
Bottrop, Germany AND e:fs TechHub GmbH, Gaimersheim, Germany.

This Source Code Form is subject to the terms of the Apache License 2.0.
If a copy of the APL2 was not distributed with this file, You can obtain
one at <https://www.apache.org/licenses/LICENSE-2.0.txt>.

**Important: updated references!** If you use this framework
(*classification or detection*) or parts of it for your research, please
cite it by:

```
@InProceedings{Kueppers_2020_CVPR_Workshops,
   author = {Küppers, Fabian and Kronenberger, Jan and Shantia, Amirhossein and Haselhoff, Anselm},
   title = {Multivariate Confidence Calibration for Object Detection},
   booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
   month = {June},
   year = {2020}
}
```

*If you use Bayesian calibration methods with uncertainty, please cite
it by*:

```
@InProceedings{Kueppers_2021_IV,
   author = {Küppers, Fabian and Kronenberger, Jan and Schneider, Jonas and Haselhoff, Anselm},
   title = {Bayesian Confidence Calibration for Epistemic Uncertainty Modelling},
   booktitle = {Proceedings of the IEEE Intelligent Vehicles Symposium (IV)},
   month = {July},
   year = {2021},
}
```

*If you use Regression calibration methods, please cite it by*:

```
@InProceedings{Kueppers_2022_ECCV_Workshops,
  author    = {Küppers, Fabian and Schneider, Jonas and Haselhoff, Anselm},
  title     = {Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection},
  booktitle = {European Conference on Computer Vision (ECCV) Workshops},
  year      = {2022},
  month     = {October},
  publisher = {Springer},
}
```

## Table of Contents

- [Overview](#overview)
  - [Update on version 1.3](#update-on-version-13)
  - [Update on version 1.2](#update-on-version-12)
  - [Update on version 1.1](#update-on-version-11)
- [Installation](#installation)
- [Requirements](#requirements)
- [Calibration Metrics](#calibration-metrics)
  - [Confidence Calibration Metrics](#confidence-calibration-metrics)
  - [Regression Calibration Metrics](#regression-calibration-metrics)
- [Methods](#methods)
  - [Confidence Calibration Methods](#confidence-calibration-methods)
    - [Binning](#binning)
    - [Scaling](#scaling)
    - [Regularization](#regularization)
  - [Regression Calibration Methods](#regression-calibration-methods)
    - [Non-parametric calibration](#non-parametric-calibration)
    - [Parametric calibration](#parametric-calibration)
- [Visualization](#visualization)
- [Examples](#examples)
  - [Classification](#classification)
    - [Post-hoc Calibration for Classification](#post-hoc-calibration-for-classification)
    - [Measuring Miscalibration for Classification](#measuring-miscalibration-for-classification)
    - [Visualizing Miscalibration for Classification](#visualizing-miscalibration-for-classification)
  - [Detection (Confidence of Objects)](#detection-confidence-of-objects)
    - [Post-hoc Calibration for Detection](#post-hoc-calibration-for-detection)
    - [Measuring Miscalibration for Detection](#measuring-miscalibration-for-detection)
    - [Visualizing Miscalibration for Detection](#visualizing-miscalibration-for-detection)
  - [Uncertainty in Confidence Calibration](#uncertainty-in-confidence-calibration)
    - [Post-hoc Calibration with Uncertainty](#post-hoc-calibration-with-uncertainty)
    - [Measuing Miscalibration with Uncertainty](#measuring-miscalibration-with-uncertainty)
  - [Probabilistic Regression](#probabilistic-regression)
    - [Post-hoc Calibration (Parametric)](#post-hoc-calibration-parametric)
    - [Post-hoc Calibration (Non-Parametric)](#post-hoc-calibration-non-parametric)
    - [Correlation Estimation and Recalibration](#correlation-estimation-and-recalibration)
    - [Measuring Miscalibration for Regression](#measuring-miscalibration-for-regression)
    - [Visualizing Miscalibration for Regression](#visualizing-miscalibration-for-regression)
- [References](#references)

## Overview

This framework is designed to calibrate the confidence estimates of
classifiers like neural networks. Modern neural networks are likely to
be overconfident with their predictions. However, reliable confidence
estimates of such classifiers are crucial especially in safety-critical
applications.

For example: given 100 predictions with a confidence of 80% of each
prediction, the observed accuracy should also match 80% (neither more
nor less). This behaviour is achievable with several calibration
methods.

### Update on version 1.3

TL;DR:

- Regression calibration methods: train and infer methods to rescale the uncertainty of probabilistic regression models
- New package: *netcal.regression* with regression calibration methods:   
  - Isotonic Regression (*netcal.regression.IsotonicRegression*)
  - Variance Scaling (*netcal.regression.VarianceScaling*)
  - GP-Beta (*netcal.regression.GPBeta*)
  - GP-Normal (*netcal.regression.GPNormal*)
  - GP-Cauchy (*netcal.regression.GPCauchy*)
- Implement *netcal.regression.GPNormal* method with correlation estimation and recalibration
- Restructured *netcal.metrics* package to distinguish between (semantic) confidence calibration in *netcal.confidence* and regression uncertainty calibration in *netcal.regression*:
  - Expected Calibration Error (ECE - *netcal.confidence.ECE*)
  - Maximum Calibration Error (MCE - *netcal.confidence.MCE*)
  - Average Calibration Error (ACE - *netcal.confidence.ACE*)
  - Maximum Mean Calibration Error (MMCE - *netcal.confidence.MMCE*)
  - Negative Log Likelihood (NLL - *netcal.regression.NLL*)
  - Prediction Interval Coverage Probability (PICP - *netcal.regression.PICP*)
  - Pinball loss (*netcal.regression.PinballLoss*)
  - Uncertainty Calibration Error (UCE - *netcal.regression.UCE*)
  - Expected Normalized Calibration Error (ENCE - *netcal.regression.ENCE*)
  - Quantile Calibration Error (QCE - *netcal.regression.QCE*)

- Added new types of reliability diagrams to visualize regression calibration properties:   
  - Reliability Regression diagram to visualize calibration for different quantile levels (preferred - *netcal.presentation.ReliabilityRegression*)
  - Reliability QCE diagram to visualize QCE over stddev (*netcal.presentation.QCE*)
- Updated examples
- Minor bugfixes
- Use library [tikzplotlib](https://github.com/texworld/tikzplotlib) within the *netcal.presentation* package to enable a direct conversion of *matplotlib.Figure* objects to Tikz-Code (e.g., can be used for LaTeX figures)

Within this release, we provide a new package *netcal.regression* to
enable recalibration of probabilistic regression tasks. Within
probabilistic regression, a regression model does not output a single
score for each prediction but rather a probability distribution (e.g.,
Gaussian with mean/variance) that targets the true output score. Similar
to (semantic) confidence calibration, regression calibration requires
that the estimated uncertainty matches the observed error distribution.
There exist several definitions for regression calibration which the
provided calibration methods aim to mitigate (cf. README within the
*netcal.regression* package). We distinguish the provided calibration
methods into non-parametric and parametric methods. Non-parametric
calibration methods take a probability distribution as input and apply
recalibration in terms of quantiles on the cumulative (CDF). This leads
to a recalibrated probability distribution that, however, has no
analytical representation but is given by certain points defining a CDF
distribution. Non-parametric calibration methods are
*netcal.regression.IsotonicRegression* and *netcal.regression.GPBeta*.

In contrast, parametric calibration methods also take a probability
distribution as input and provide a recalibrated distribution that has
an analytical expression (e.g., Gaussian). Parametric calibration
methods are *netcal.regression.VarianceScaling*,
*netcal.regression.GPNormal*, and *netcal.regression.GPCauchy*.

The calibration methods are designed to also work with multiple
independent dimensions. The methods
*netcal.regression.IsotonicRegression* and
*netcal.regression.VarianceScaling* apply a recalibration of each
dimension independently of each other. In contrast, the GP methods
*netcal.regression.GPBeta*, *netcal.regression.GPNormal*, and
*netcal.regression.GPCauchy* use a single GP to apply recalibration.
Furthermore, the GP-Normal *netcal.regression.GPNormal* is can model
possible correlations within the training data to transform multiple
univariate probability distributions of a single sample to a joint
multivariate (normal) distribution with possible correlations. This
calibration scheme is denoted as *correlation estimation*. Additionally,
the GP-Normal is also able to take a multivariate (normal) distribution
with correlations as input and applies a recalibration of the whole
covariance matrix. This is referred to as *correlation recalibration*.

Besides the recalibration methods, we restructured the *netcal.metrics*
package which now also holds several metrics for regression calibration
(cf. *netcal.metrics* package documentation for detailed information).
Finally, we provide several ways to visualize regression miscalibration
within the *netcal.presentation* package.

All plot-methods within the *netcal.presentation* package now support
the option "tikz=True" which switches from standard
*matplotlib.Figure* objects to strings with Tikz-Code. Tikz-code can be
directly used for LaTeX documents to render images as vector graphics
with high quality. Thus, this option helps to improve the quality of
your reliability diagrams if you are planning to use this library for
any type of publication/document

### Update on version 1.2

TL;DR:

- Bayesian confidence calibration: train and infer scaling methods using variational inference (VI) and MCMC sampling
- New metrics: MMCE [[13]](#ref13) and PICP [[14]](#ref14) (*netcal.metrics.MMCE* and *netcal.metrics.PICP*)
- New regularization methods: MMCE [[13]](#ref13) and DCA [[15]](#ref15) (*netcal.regularization.MMCEPenalty* and *netcal.regularization.DCAPenalty*)
- Updated examples
- Switched license from MPL2 to APL2

Now you can also use Bayesian methods to obtain uncertainty within a
calibration mapping mainly in the *netcal.scaling* package. We adapted
Markov-Chain Monte-Carlo sampling (MCMC) as well as Variational
Inference (VI) on common calibration methods. It is also easily possible
to bring the scaling methods to CUDA in order to speed-up the
computations. We further provide new metrics to evaluate confidence
calibration (MMCE) and to evaluate the quality of prediction intervals
(PICP). Finally, we updated our framework by new regularization methods
that can be used during model training (MMCE and DCA).

### Update on version 1.1

This framework can also be used to calibrate object detection models. It
has recently been shown that calibration on object detection also
depends on the position and/or scale of a predicted object [[12]](#ref12). We
provide calibration methods to perform confidence calibration w.r.t. the
additional box regression branch. For this purpose, we extended the
commonly used Histogram Binning [[3]](#ref3), Logistic Calibration alias Platt
scaling [[10]](#ref10) and the Beta Calibration method [[2]](#ref2) to also include the
bounding box information into a calibration mapping. Furthermore, we
provide two new methods called the *Dependent Logistic Calibration* and
the *Dependent Beta Calibration* that are not only able to perform a
calibration mapping w.r.t. additional bounding box information but also
to model correlations and dependencies between all given quantities [[12]](#ref12).
Those methods should be preffered over their counterparts in object
detection mode.

The framework is structured as follows:

    netcal
      .binning         # binning methods (confidence calibration)
      .scaling         # scaling methods (confidence calibration)
      .regularization  # regularization methods (confidence calibration)
      .presentation    # presentation methods (confidence/regression calibration)
      .metrics         # metrics for measuring miscalibration (confidence/regression calibration)
      .regression      # methods for regression uncertainty calibration (regression calibration)

    examples           # example code snippets

## Installation

The installation of the calibration suite is quite easy as it registered
in the Python Package Index (PyPI). You can either install this
framework using PIP:
```shell
$ python3 -m pip install netcal
```
Or simply invoke the following command to install the calibration suite when installing from source:
```shell
$ git clone https://github.com/EFS-OpenSource/calibration-framework
$ cd calibration-framework
$ python3 -m pip install .
```

Note: with update 1.3, we switched from *setup.py* to *pyproject.toml*
according to PEP-518. The *setup.py* is only for backwards
compatibility.

### Requirements
According to *requierments.txt*:

-   numpy\>=1.18
-   scipy\>=1.4
-   matplotlib\>=3.3
-   scikit-learn\>=0.24
-   torch\>=1.9
-   torchvision\>=0.10.0
-   tqdm\>=4.40
-   pyro-ppl\>=1.8
-   tikzplotlib\>=0.9.8
-   tensorboard\>=2.2
-   gpytorch\>=1.5.1

## Calibration Metrics

We further distinguish between *onfidence calibration* which aims to
recalibrate confidence estimates in the [0, 1] interval, and
*regression uncertainty calibration* which addresses the problem of
calibration in probabilistic regression settings.

### Confidence Calibration Metrics

The most common metric to determine miscalibration in the scope of
classification is the *Expected Calibration Error* (ECE) [[1]](#ref1). This
metric divides the confidence space into several bins and measures the
observed accuracy in each bin. The bin gaps between observed accuracy
and bin confidence are summed up and weighted by the amount of samples
in each bin. The *Maximum Calibration Error* (MCE) denotes the highest
gap over all bins. The *Average Calibration Error* (ACE) [[11]](#ref11) denotes
the average miscalibration where each bin gets weighted equally. For
object detection, we implemented the *Detection Calibration Error*
(D-ECE) [[12]](#ref12) that is the natural extension of the ECE to object
detection tasks. The miscalibration is determined w.r.t. the bounding
box information provided (e.g. box location and/or scale). For this
purpose, all available information gets binned in a multidimensional
histogram. The accuracy is then calculated in each bin separately to
determine the mean deviation between confidence and accuracy.

- (Detection) Expected Calibration Error [[1]](#ref1), [[12]](#ref12) (*netcal.metrics.ECE*)
- (Detection) Maximum Calibration Error [[1]](#ref1), [[12]](#ref12)  (*netcal.metrics.MCE*)
- (Detection) Average Calibration Error [[11]](#ref11), [[12]](#ref12) (*netcal.metrics.ACE*)
- Maximum Mean Calibration Error (MMCE) [[13]](#ref13) (*netcal.metrics.MMCE*) (no position-dependency)

### Regression Calibration Metrics

In regression calibration, the most common metric is the *Negative Log
Likelihood* (NLL) to measure the quality of a predicted probability
distribution w.r.t. the ground-truth:

- Negative Log Likelihood (NLL) (*netcal.metrics.NLL*)

The metrics *Pinball Loss*, *Prediction Interval Coverage Probability*
(PICP), and *Quantile Calibration Error* (QCE) evaluate the estimated
distributions by means of the predicted quantiles. For example, if a
forecaster makes 100 predictions using a probability distribution for
each estimate targeting the true ground-truth, we can measure the
coverage of the ground-truth samples for a certain quantile level (e.g.,
95% quantile). If the relative amount of ground-truth samples falling
into a certain predicted quantile is above or below the specified
quantile level, a forecaster is told to be miscalibrated in terms of
*quantile calibration*. Appropriate metrics in this context are

- Pinball Loss (*netcal.metrics.PinballLoss*)
- Prediction Interval Coverage Probability (PICP) [[14]](#ref14) (*netcal.metrics.PICP*)
- Quantile Calibration Error (QCE) [[15]](#ref15) (*netcal.metrics.QCE*)

Finally, if we work with normal distributions, we can measure the
quality of the predicted variance/stddev estimates. For *variance
calibration*, it is required that the predicted variance mathes the
observed error variance which is equivalent to then Mean Squared Error
(MSE). Metrics for *variance calibration* are

- Expected Normalized Calibration Error (ENCE) [[17]](#ref17) (*netcal.metrics.ENCE*)
- Uncertainty Calibration Error (UCE) [[18]](#ref18) (*netcal.metrics.UCE*)

## Methods

We further give an overview about the post-hoc calibration methods for
(semantic) confidence calibration as well as about the methods for
regression uncertainty calibration.

### Confidence Calibration Methods

The post-hoc calibration methods are separated into binning and scaling
methods. The binning methods divide the available information into
several bins (like ECE or D-ECE) and perform calibration on each bin.
The scaling methods scale the confidence estimates or logits directly to
calibrated confidence estimates - on detection calibration, this is done
w.r.t. the additional regression branch of a network.

Important: if you use the detection mode, you need to specifiy the flag
"detection=True" in the constructor of the according method (this is
not necessary for *netcal.scaling.LogisticCalibrationDependent* and
*netcal.scaling.BetaCalibrationDependent*).

Most of the calibration methods are designed for binary classification
tasks. For binning methods, multi-class calibration is performed in
"one vs. all" by default.

Some methods such as "Isotonic Regression" utilize methods from the
scikit-learn API [[9]](#ref9).

Another group are the regularization tools which are added to the loss
during the training of a Neural Network.

#### Binning

Implemented binning methods are:

- Histogram Binning for classification [[3]](#ref3), [[4]](#ref4) and object detection [[12]](#ref12) (*netcal.binning.HistogramBinning*)
- Isotonic Regression [[4]](#ref4),[[5]](#ref5) (*netcal.binning.IsotonicRegression*)
- Bayesian Binning into Quantiles (BBQ) [[1]](#ref1) (*netcal.binning.BBQ*)
- Ensemble of Near Isotonic Regression (ENIR) [[6]](#ref6) (*netcal.binning.ENIR*)

#### Scaling

Implemented scaling methods are:

- Logistic Calibration/Platt Scaling for classification [[10]](#ref10) and object detection [[12]](#ref12) (*netcal.scaling.LogisticCalibration*)
- Dependent Logistic Calibration for object detection [[12]](#ref12) (*netcal.scaling.LogisticCalibrationDependent*) - on detection, this method is able to capture correlations between all input quantities and should be preferred over Logistic Calibration for object detection
- Temperature Scaling for classification [[7]](#ref7) and object detection [[12]](#ref12) (*netcal.scaling.TemperatureScaling*)
- Beta Calibration for classification [[2]](#ref2) and object detection [[12]](#ref12) (*netcal.scaling.BetaCalibration*)
- Dependent Beta Calibration for object detection [[12]](#ref12) (*netcal.scaling.BetaCalibrationDependent*) - on detection, this method is able to capture correlations between all input quantities and should be preferred over Beta Calibration for object detection

**New on version 1.2:** you can provide a parameter named "method" to
the constructor of each scaling method. This parameter could be one of
the following: - 'mle': use the method feed-forward with maximum
likelihood estimates on the calibration parameters (standard) -
'momentum': use non-convex momentum optimization (e.g. default on
dependent beta calibration) - 'mcmc': use Markov-Chain Monte-Carlo
sampling to obtain multiple parameter sets in order to quantify
uncertainty in the calibration - 'variational': use Variational
Inference to obtain multiple parameter sets in order to quantify
uncertainty in the calibration

#### Regularization

With some effort, it is also possible to push the model training towards
calibrated confidences by regularization. Implemented regularization
methods are:

- Confidence Penalty [[8]](#ref8) (*netcal.regularization.confidence\_penalty* and *netcal.regularization.ConfidencePenalty* - the latter one is a PyTorch implementation that might be used as a regularization term)
- Maximum Mean Calibration Error (MMCE) [[13]](#ref13) (*netcal.regularization.MMCEPenalty* - PyTorch regularization module)
- DCA [[15]](#ref15) (*netcal.regularization.DCAPenalty* - PyTorch regularization module)

### Regression Calibration Methods

The *netcal* library provides post-hoc methods to recalibrate the
uncertainty of probabilistic regression tasks. We distinguish the
calibration methods into non-parametric and parametric methods.
Non-parametric calibration methods take a probability distribution as
input and apply recalibration in terms of quantiles on the cumulative
(CDF). This leads to a recalibrated probability distribution that,
however, has no analytical representation but is given by certain points
defining a CDF distribution. In contrast, parametric calibration methods
also take a probability distribution as input and provide a recalibrated
distribution that has an analytical expression (e.g., Gaussian).

#### Non-parametric calibration

The common non-parametric recalibration methods use the predicted
cumulative (CDF) distribution functions to learn a mapping from the
uncalibrated quantiles to the observed quantile coverage. Using a
recalibrated CDF, it is possible to derive the respective density
functions (PDF) or to extract statistical moments such as mean and
variance. Non-parametric calibration methods within the
*netcal.regression* package are

- Isotonic Regression [[19]](#ref19) which applies a (marginal) recalibration of the CDF (*netcal.regression.IsotonicRegression*)
- GP-Beta [[20]](#ref20) which applies an input-dependent recalibration of the CDF using a Gaussian process for parameter estimation (*netcal.regression.GPBeta*)

#### Parametric calibration

The parametric recalibration methods apply a recalibration of the
estimated distributions so that the resulting distribution is given in
terms of a distribution with an analytical expression (e.g., a
Gaussian). These methods are suitable for applications where a
parametric distribution is required for subsequent applications, e.g.,
within Kalman filtering. We implemented the following parametric
calibration methods:

- Variance Scaling [[17]](#ref17), [[18]](#ref18) which is nothing else but a temperature scaling for the predicted variance (*netcal.regression.VarianceScaling*)
- GP-Normal [[16]](#ref16) which applies an input-dependent rescaling of the predicted variance (*netcal.regression.GPNormal*). Note: this method is also able to capture correlations between multiple input dimensions and can return a joint multivariate normal distribution as calibration output (cf. examples section).
- GP-Cauchy [[16]](#ref16) is similar to GP-Normal but utilizes a Cauchy distribution as calibration output (*netcal.regression.GPCauchy*)

## Visualization

For visualization of miscalibration, one can use a Confidence Histograms
& Reliability Diagrams for (semantic) confidence calibration as well as
for regression uncertainty calibration. Within confidence calibration,
these diagrams are similar to ECE. The output space is divided into
equally spaced bins. The calibration gap between bin accuracy and bin
confidence is visualized as a histogram.

For detection calibration, the miscalibration can be visualized either
along one additional box information (e.g. the x-position of the
predictions) or distributed over two additional box information in terms
of a heatmap.

For regression uncertainty calibration, the reliability diagram shows
the relative prediction interval coverage of the ground-truth samples
for different quantile levels.

- Reliability Diagram [[1]](#ref1), [[12]](#ref12) (*netcal.presentation.ReliabilityDiagram*)
- Reliability Diagram for regression calibration (*netcal.presentation.ReliabilityRegression*)
- Reliability QCE Diagram [[16]](#ref16) shows the Quantile Calibration Error (QCE) for different variance levels (*netcal.presentation.ReliabilityQCE*)

**New on version 1.3:** All plot-methods within the
*netcal.presentation* package now support the option "tikz=True" which
switches from standard *matplotlib.Figure* objects to strings with
Tikz-Code. Tikz-code can be directly used for LaTeX documents to render
images as vector graphics with high quality. Thus, this option helps to
improve the quality of your reliability diagrams if you are planning to
use this library for any type of publication/document

## Examples

The calibration methods work with the predicted confidence estimates of
a neural network and on detection also with the bounding box regression
branch.

### Classification

This is a basic example which uses softmax predictions of a
classification task with 10 classes and the given NumPy arrays:

```python
ground_truth  # this is a NumPy 1-D array with ground truth digits between 0-9 - shape: (n_samples,)
confidences   # this is a NumPy 2-D array with confidence estimates between 0-1 - shape: (n_samples, n_classes)
```

#### Post-hoc Calibration for Classification

This is an example for *netcal.scaling.TemperatureScaling* but also
works for every calibration method (remind different constructor
parameters):

```python
import numpy as np
from netcal.scaling import TemperatureScaling

temperature = TemperatureScaling()
temperature.fit(confidences, ground_truth)
calibrated = temperature.transform(confidences)
```

#### Measuring Miscalibration for Classification

The miscalibration can be determined with the ECE:

```python
from netcal.metrics import ECE

n_bins = 10

ece = ECE(n_bins)
uncalibrated_score = ece.measure(confidences, ground_truth)
calibrated_score = ece.measure(calibrated, ground_truth)
```

#### Visualizing Miscalibration for Classification

The miscalibration can be visualized with a Reliability Diagram:

```python
from netcal.presentation import ReliabilityDiagram

n_bins = 10

diagram = ReliabilityDiagram(n_bins)
diagram.plot(confidences, ground_truth)  # visualize miscalibration of uncalibrated
diagram.plot(calibrated, ground_truth)   # visualize miscalibration of calibrated

# you can also use this method to create a tikz file with tikz code
# that can be directly used within LaTeX documents:
diagram.plot(confidences, ground_truth, tikz=True, filename="diagram.tikz")
```

### Detection (Confidence of Objects)

In this example we use confidence predictions of an object detection
model with the according x-position of the predicted bounding boxes. Our
ground-truth provided to the calibration algorithm denotes if a bounding
box has matched a ground-truth box with a certain IoU and the correct
class label.

```python
matched                # binary NumPy 1-D array (0, 1) that indicates if a bounding box has matched a ground truth at a certain IoU with the right label - shape: (n_samples,)
confidences            # NumPy 1-D array with confidence estimates between 0-1 - shape: (n_samples,)
relative_x_position    # NumPy 1-D array with relative center-x position between 0-1 of each prediction - shape: (n_samples,)
```

#### Post-hoc Calibration for Detection

This is an example for *netcal.scaling.LogisticCalibration* and
*netcal.scaling.LogisticCalibrationDependent* but also works for every
calibration method (remind different constructor parameters):

```python
import numpy as np
from netcal.scaling import LogisticCalibration, LogisticCalibrationDependent

input = np.stack((confidences, relative_x_position), axis=1)

lr = LogisticCalibration(detection=True, use_cuda=False)    # flag 'detection=True' is mandatory for this method
lr.fit(input, matched)
calibrated = lr.transform(input)

lr_dependent = LogisticCalibrationDependent(use_cuda=False) # flag 'detection=True' is not necessary as this method is only defined for detection
lr_dependent.fit(input, matched)
calibrated = lr_dependent.transform(input)
```

#### Measuring Miscalibration for Detection

The miscalibration can be determined with the D-ECE:

```python
from netcal.metrics import ECE

n_bins = [10, 10]
input_calibrated = np.stack((calibrated, relative_x_position), axis=1)

ece = ECE(n_bins, detection=True)           # flag 'detection=True' is mandatory for this method
uncalibrated_score = ece.measure(input, matched)
calibrated_score = ece.measure(input_calibrated, matched)
```

#### Visualizing Miscalibration for Detection

The miscalibration can be visualized with a Reliability Diagram:

```python
from netcal.presentation import ReliabilityDiagram

n_bins = [10, 10]

diagram = ReliabilityDiagram(n_bins, detection=True)    # flag 'detection=True' is mandatory for this method
diagram.plot(input, matched)                # visualize miscalibration of uncalibrated
diagram.plot(input_calibrated, matched)     # visualize miscalibration of calibrated

# you can also use this method to create a tikz file with tikz code
# that can be directly used within LaTeX documents:
diagram.plot(input, matched, tikz=True, filename="diagram.tikz")
```

### Uncertainty in Confidence Calibration

We can also quantify the uncertainty in a calibration mapping if we use
a Bayesian view on the calibration models. We can sample multiple
parameter sets using MCMC sampling or VI. In this example, we reuse the
data of the previous detection example.

```python
matched                # binary NumPy 1-D array (0, 1) that indicates if a bounding box has matched a ground truth at a certain IoU with the right label - shape: (n_samples,)
confidences            # NumPy 1-D array with confidence estimates between 0-1 - shape: (n_samples,)
relative_x_position    # NumPy 1-D array with relative center-x position between 0-1 of each prediction - shape: (n_samples,)
```

#### Post-hoc Calibration with Uncertainty

This is an example for *netcal.scaling.LogisticCalibration* and
*netcal.scaling.LogisticCalibrationDependent* but also works for every
calibration method (remind different constructor parameters):

```python
import numpy as np
from netcal.scaling import LogisticCalibration, LogisticCalibrationDependent

input = np.stack((confidences, relative_x_position), axis=1)

# flag 'detection=True' is mandatory for this method
# use Variational Inference with 2000 optimization steps for creating this calibration mapping
lr = LogisticCalibration(detection=True, method'variational', vi_epochs=2000, use_cuda=False)
lr.fit(input, matched)

# 'num_samples=1000': sample 1000 parameter sets from VI
# thus, 'calibrated' has shape [1000, n_samples]
calibrated = lr.transform(input, num_samples=1000)

# flag 'detection=True' is not necessary as this method is only defined for detection
# this time, use Markov-Chain Monte-Carlo sampling with 250 warm-up steps, 250 parameter samples and one chain
lr_dependent = LogisticCalibrationDependent(method='mcmc',
                                            mcmc_warmup_steps=250, mcmc_steps=250, mcmc_chains=1,
                                            use_cuda=False)
lr_dependent.fit(input, matched)

# 'num_samples=1000': although we have only sampled 250 different parameter sets,
# we can randomly sample 1000 parameter sets from MCMC
calibrated = lr_dependent.transform(input)
```

#### Measuring Miscalibration with Uncertainty

You can directly pass the output to the D-ECE and PICP instance to
measure miscalibration and mask quality:

```python
from netcal.metrics import ECE
from netcal.metrics import PICP

n_bins = 10
ece = ECE(n_bins, detection=True)
picp = PICP(n_bins, detection=True)

# the following function calls are equivalent:
miscalibration = ece.measure(calibrated, matched, uncertainty="mean")
miscalibration = ece.measure(np.mean(calibrated, axis=0), matched)

# now determine uncertainty quality
uncertainty = picp.measure(calibrated, matched, kind="confidence")

print("D-ECE:", miscalibration)
print("PICP:", uncertainty.picp) # prediction coverage probability
print("MPIW:", uncertainty.mpiw) # mean prediction interval width
```

If we want to measure miscalibration and uncertainty quality by means of
the relative x position, we need to broadcast the according information:

```python
# broadcast and stack x information to calibrated information
broadcasted = np.broadcast_to(relative_x_position, calibrated.shape)
calibrated = np.stack((calibrated, broadcasted), axis=2)

n_bins = [10, 10]
ece = ECE(n_bins, detection=True)
picp = PICP(n_bins, detection=True)

# the following function calls are equivalent:
miscalibration = ece.measure(calibrated, matched, uncertainty="mean")
miscalibration = ece.measure(np.mean(calibrated, axis=0), matched)

# now determine uncertainty quality
uncertainty = picp.measure(calibrated, matched, uncertainty="mean")

print("D-ECE:", miscalibration)
print("PICP:", uncertainty.picp) # prediction coverage probability
print("MPIW:", uncertainty.mpiw) # mean prediction interval width
```

### Probabilistic Regression

The following example shows how to use the post-hoc calibration methods
for probabilistic regression tasks. Within probabilistic regression, a
forecaster (e.g. with Gaussian prior) outputs a mean and a variance
targeting the true ground-truth score. Thus, the following information
is required to construct the calibration methods:

```python
mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions
```

#### Post-hoc Calibration (Parametric)

These information might result e.g. from object detection where the
position information of the objects (bounding boxes) are parametrized by
normal distributions. We start by using parametric calibration methods
such as Variance Scaling:

```python
from netcal.regression import VarianceScaling, GPNormal

# the initialization of the Variance Scaling method is pretty simple
varscaling = VarianceScaling()

# the GP-Normal requires a little bit more parameters to parametrize the underlying GP
gpnormal = GPNormal(
    n_inducing_points=12,    # number of inducing points
    n_random_samples=256,    # random samples used for likelihood
    n_epochs=256,            # optimization epochs
    use_cuda=False,          # can also use CUDA for computations
)

# fit the Variance Scaling
# note that we need to pass the first argument as tuple as the input distributions
# are parametrized by mean and variance
varscaling.fit((mean, stddev), ground_truth)

# fit GP-Normal - similar parameters here!
gpnormal.fit((mean, stddev), ground_truth)

# transform distributions to obtain recalibrated stddevs
stddev_varscaling = varscaling.transform((mean, stddev))  # NumPy array with stddev - has shape (n, d)
stddev_gpnormal = gpnormal.transform((mean, stddev))  # NumPy array with stddev - has shape (n, d)
```

#### Post-hoc Calibration (Non-Parametric)

We can also use non-parametric calibration methods. In this case, the
calibrated distributions are defined by their density (PDF) and
cumulative (CDF) functions:

```python
from netcal.regression import IsotonicRegression, GPBeta

# the initialization of the Isotonic Regression method is pretty simple
isotonic = IsotonicRegression()

# the GP-Normal requires a little bit more parameters to parametrize the underlying GP
gpbeta = GPBeta(
    n_inducing_points=12,    # number of inducing points
    n_random_samples=256,    # random samples used for likelihood
    n_epochs=256,            # optimization epochs
    use_cuda=False,          # can also use CUDA for computations
)

# fit the Isotonic Regression
# note that we need to pass the first argument as tuple as the input distributions
# are parametrized by mean and variance
isotonic.fit((mean, stddev), ground_truth)

# fit GP-Beta - similar parameters here!
gpbeta.fit((mean, stddev), ground_truth)

# transform distributions to obtain recalibrated distributions
t_isotonic, pdf_isotonic, cdf_isotonic = varscaling.transform((mean, stddev))
t_gpbeta, pdf_gpbeta, cdf_gpbeta = gpbeta.transform((mean, stddev))

# Note: the transformation results are NumPy n-d arrays with shape (t, n, d)
# with t as the number of points that define the PDF/CDF,
# with n as the number of samples, and
# with d as the number of dimensions.

# The resulting variables can be interpreted as follows:
# - t_isotonic/t_gpbeta: x-values of the PDF/CDF with shape (t, n, d)
# - pdf_isotonic/pdf_gpbeta: y-values of the PDF with shape (t, n, d)
# - cdf_isotonic/cdf_gpbeta: y-values of the CDF with shape (t, n, d)
```

You can visualize the non-parametric distribution of a single sample
within a single dimension using Matplotlib:

```python
from matplotlib import pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1)

# plot the recalibrated PDF within a single axis after calibration
ax1.plot(
    t_isotonic[:, 0, 0], pdf_isotonic[:, 0, 0],
    t_gpbeta[:, 0, 0], pdf_gpbeta[:, 0, 0],
)

# plot the recalibrated PDF within a single axis after calibration
ax2.plot(
    t_isotonic[:, 0, 0], cdf_isotonic[:, 0, 0],
    t_gpbeta[:, 0, 0], cdf_gpbeta[:, 0, 0],
)

plt.show()
```

We provide a method to extract the statistical moments expectation and
variance from the recalibrated cumulative (CDF). Note that we advise to
use one of the parametric calibration methods if you need e.g. a
Gaussian for subsequent applications such as Kalman filtering.

```python
from netcal import cumulative_moments

# extract the expectation (mean) and the variance from the recalibrated CDF
ymean_isotonic, yvar_isotonic = cumulative_moments(t_isotonic, cdf_isotonic)
ymean_gpbeta, yvar_gpbeta = cumulative_moments(t_gpbeta, cdf_gpbeta)

# each of these variables has shape (n, d) and holds the
# mean/variance for each sample and in each dimension
```

#### Correlation Estimation and Recalibration

With the GP-Normal *netcal.regression.GPNormal*, it is also possible to
detect possible correlations between multiple input dimensions that have
originally been trained/modelled independently from each other:

```python
from netcal.regression import GPNormal

# the GP-Normal requires a little bit more parameters to parametrize the underlying GP
gpnormal = GPNormal(
    n_inducing_points=12,    # number of inducing points
    n_random_samples=256,    # random samples used for likelihood
    n_epochs=256,            # optimization epochs
    use_cuda=False,          # can also use CUDA for computations
    correlations=True,       # enable correlation capturing between the input dimensions
)

# fit GP-Normal
# note that we need to pass the first argument as tuple as the input distributions
# are parametrized by mean and variance
gpnormal.fit((mean, stddev), ground_truth)

# transform distributions to obtain recalibrated covariance matrices
cov = gpnormal.transform((mean, stddev))  # NumPy array with covariance - has shape (n, d, d)

# note: if the input is already given by multivariate normal distributions
# (stddev is covariance and has shape (n, d, d)), the methods works similar
# and simply applies a covariance recalibration of the input
```

#### Measuring Miscalibration for Regression

Measuring miscalibration is as simple as the training of the methods:

```python
import numpy as np
from netcal.metrics import NLL, PinballLoss, QCE

# define the quantile levels that are used to evaluate the pinball loss and the QCE
quantiles = np.linspace(0.1, 0.9, 9)

# initialize NLL, Pinball, and QCE objects
nll = NLL()
pinball = PinballLoss()
qce = QCE(marginal=True)  # if "marginal=False", we can also measure the QCE by means of the predicted variance levels (realized by binning the variance space)

# measure miscalibration with the initialized metrics
# Note: the parameter "reduction" has a major influence to the return shape of the metrics
# see the method docstrings for detailed information
nll.measure((mean, stddev), ground_truth, reduction="mean")
pinball.measure((mean, stddev), ground_truth, q=quantiles, reduction="mean")
qce.measure((mean, stddev), ground_truth, q=quantiles, reduction="mean")
```

#### Visualizing Miscalibration for Regression

Example visualization code block using the
*netcal.presentation.ReliabilityRegression* class:

```python
from netcal.presentation import ReliabilityRegression

# define the quantile levels that are used for the quantile evaluation
quantiles = np.linspace(0.1, 0.9, 9)

# initialize the diagram object
diagram = ReliabilityRegression(quantiles=quantiles)

# visualize miscalibration with the initialized object
diagram.plot((mean, stddev), ground_truth)

# you can also use this method to create a tikz file with tikz code
# that can be directly used within LaTeX documents:
diagram.plot((mean, stddev), ground_truth, tikz=True, filename="diagram.tikz")
```
</div>

## References

<a name="ref1">[1]</a> Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht: "Obtaining well calibrated probabilities using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.

<a name="ref2">[2]</a> Kull, Meelis, Telmo Silva Filho, and Peter Flach: "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers." Artificial Intelligence and Statistics, PMLR 54:623-631, 2017.

<a name="ref3">[3]</a> Zadrozny, Bianca and Elkan, Charles: "Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers." In ICML, pp. 609–616, 2001.

<a name="ref4">[4]</a> Zadrozny, Bianca and Elkan, Charles: "Transforming classifier scores into accurate multiclass probability estimates." In KDD, pp. 694–699, 2002.

<a name="ref5">[5]</a> Ryan J Tibshirani, Holger Hoefling, and Robert Tibshirani: "Nearly-isotonic regression." Technometrics, 53(1):54–61, 2011.

<a name="ref6">[6]</a> Naeini, Mahdi Pakdaman, and Gregory F. Cooper: "Binary classifier calibration using an ensemble of near isotonic regression models." 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.

<a name="ref7">[7]</a> Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger: "On Calibration of Modern Neural Networks." Proceedings of the 34th International Conference on Machine Learning, 2017.

<a name="ref8">[8]</a> Pereyra, G., Tucker, G., Chorowski, J., Kaiser, L. and Hinton, G.: “Regularizing neural networks by penalizing confident output distributions.” CoRR, 2017.

<a name="ref9">[9]</a> Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E.: "Scikit-learn: Machine Learning in Python." In Journal of Machine Learning Research, volume 12 pp 2825-2830, 2011.

<a name="ref10">[10]</a> Platt, John: "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." Advances in large margin classifiers, 10(3): 61–74, 1999.

<a name="ref11">[11]</a> Neumann, Lukas, Andrew Zisserman, and Andrea Vedaldi: "Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection." Conference on Neural Information Processing Systems (NIPS) Workshop MLITS, 2018.

<a name="ref12">[12]</a> Fabian Küppers, Jan Kronenberger, Amirhossein Shantia, and Anselm Haselhoff: "Multivariate Confidence Calibration for Object Detection"." The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020

<a name="ref13">[13]</a> Kumar, Aviral, Sunita Sarawagi, and Ujjwal Jain: "Trainable calibration measures for neural networks from _kernel mean embeddings." International Conference on Machine Learning. 2018

<a name="ref14">[14]</a> Jiayu  Yao,  Weiwei  Pan,  Soumya  Ghosh,  and  Finale  Doshi-Velez: "Quality of Uncertainty Quantification for Bayesian Neural Network Inference." Workshop on Uncertainty and Robustness in Deep Learning, ICML, 2019

<a name="ref15">[15]</a> Liang, Gongbo, et al.: "Improved trainable calibration method for neural networks on medical imaging classification." arXiv preprint arXiv:2009.04057 (2020)

<a name="ref16">[16]</a> Fabian Küppers, Jonas Schneider, Jonas, and Anselm Haselhoff: "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection." In: Proceedings of the European Conference on Computer Vision (ECCV) Workshops, Springer, October 2022

<a name="ref17">[17]</a> Levi, Dan, et al.: "Evaluating and calibrating uncertainty prediction in regression tasks." arXiv preprint arXiv:1905.11659 (2019).

<a name="ref18">[18]</a> Laves, Max-Heinrich, et al.: "Well-calibrated regression uncertainty in medical imaging with deep learning." Medical Imaging with Deep Learning. PMLR, 2020.

<a name="ref19">[19]</a> Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon: "Accurate uncertainties for deep learning using calibrated regression." International Conference on Machine Learning. PMLR, 2018.

<a name="ref20">[20]</a> Hao Song, Tom Diethe, Meelis Kull and Peter Flach: "Distribution calibration for regression." International Conference on Machine Learning. PMLR, 2019.
