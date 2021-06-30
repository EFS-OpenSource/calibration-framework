# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from netcal.scaling import LogisticCalibration


class TemperatureScaling(LogisticCalibration):
    """
    On classification or detection, apply the temperature scaling method described in [1]_ to obtain a
    calibration mapping. For confidence calibration in classification tasks, a
    confidence mapping :math:`g` is applied on top of a miscalibrated scoring classifier :math:`\\hat{p} = h(x)` to
    deliver a calibrated confidence score :math:`\\hat{q} = g(h(x))`.

    For detection calibration, we can also use the additional box regression output which we denote as
    :math:`\\hat{r} \\in [0, 1]^J` with :math:`J` as the number of dimensions used for the box encoding (e.g.
    :math:`J=4` for x position, y position, width and height).
    Therefore, the calibration map is not only a function of the confidence score, but also of :math:`\\hat{r}`.
    To define a general calibration map, we use the the combined input :math:`s = (\\hat{p}, \\hat{r})` of size K
    and perform a temperature scaling defined by

    .. math::

       \\hat{q} = \\sigma(s / T)

    with the temperature :math:`T \\in \\mathbb{R}` as a single scalar value.
    The function :math:`\\sigma(*)` is either the sigmoid (on detection or binary classification) or the
    softmax operator (multiclass classification).

    We utilize standard optimization methods to determine the calibration mapping :math:`g(s)`.

    Parameters
    ----------
    method : str, default: "mle"
        Method that is used to obtain a calibration mapping:
        - 'mle': Maximum likelihood estimate without uncertainty using a convex optimizer.
        - 'momentum': MLE estimate using Momentum optimizer for non-convex optimization.
        - 'variational': Variational Inference with uncertainty.
        - 'mcmc': Markov-Chain Monte-Carlo sampling with uncertainty.
    momentum_epochs : int, optional, default: 1000
            Number of epochs used by momentum optimizer.
    mcmc_steps : int, optional, default: 20
        Number of weight samples obtained by MCMC sampling.
    mcmc_chains : int, optional, default: 1
        Number of Markov-chains used in parallel for MCMC sampling (this will result
        in mcmc_steps * mcmc_chains samples).
    mcmc_warmup_steps : int, optional, default: 100
        Warmup steps used for MCMC sampling.
    vi_epochs : int, optional, default: 1000
        Number of epochs used for ELBO optimization.
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).
    use_cuda : str or bool, optional, default: False
        Specify if CUDA should be used. If str, you can also specify the device
        number like 'cuda:0', etc.

    References
    ----------
    .. [1] Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
       "On Calibration of Modern Neural Networks."
       Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
       `Get source online <https://arxiv.org/abs/1706.04599>`_

    .. [2] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020.

    .. [3] Fabian Küppers, Jan Kronenberger, Jonas Schneider  and Anselm Haselhoff:
       "Bayesian Confidence Calibration for Epistemic Uncertainty Modelling."
       2021 IEEE Intelligent Vehicles Symposium (IV), 2021
    """

    def __init__(self, *args, **kwargs):
        """ Create an instance of `TemperatureScaling`. Detailed parameter description given in class docs. """

        super().__init__(*args, **kwargs)
        self.temperature_only = True

    @property
    def temperature(self):
        """ Getter for temperature of temperature scaling. """
        return self.weights
