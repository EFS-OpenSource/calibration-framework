# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
import numpy as np
from sklearn.model_selection import train_test_split

from netcal import global_accepts
from netcal.metrics import ACE, ECE, MCE
from netcal.presentation import ReliabilityDiagram


@global_accepts(list, str, int, (None, str), float, bool, str)
def single_example(models: list, datafile: str, bins: int, diagram: str = None,
                   validation_split: float = 0.7, save_models: bool = False, domain: str = ".") -> int:
    """
    Measure miscalibration of given methods on specified dataset.

    Parameters
    ----------
    models : list
        List of tuples with [('<name>', <instance of CalibrationMethod>), ...].
    datafile : str
        Path to datafile which contains two NumPy arrays with keys 'ground_truth' and 'predictions'.
    bins : int
        Number of bins used by ECE, MCE and ReliabilityDiagram.
    diagram : str, optional, default: None
        Type of diagram wich should be plotted. This could be 'diagram', 'curve', 'inference' or None.
    validation_split : float
        Split ratio between build set and validation set.
    save_models : bool
        True if instances of calibration methods should be stored.
    domain : str, optional, default: "."
        Domain/directory where to store the results.

    Returns
    -------
    int
        0 on success, -1 otherwise
    """

    if not os.path.exists(datafile):
        print("Dataset \'%s\' does not exist" % datafile)
        return -1

    # read NumPy input files
    try:
        with open(datafile, "rb") as open_file:
            npzfile = np.load(open_file)
            ground_truth = npzfile['ground_truth'].squeeze()
            predictions = npzfile['predictions'].squeeze()
    except KeyError:
        print("Key \'ground_truth\' or \'predictions\' not found in file \'%s\'" % datafile)
        return -1

    # split data set into build set and validation set
    build_set_gt, validation_set_gt, build_set_sm, validation_set_sm = train_test_split(ground_truth, predictions,
                                                                                        test_size=validation_split,
                                                                                        stratify=ground_truth,
                                                                                        random_state=None)

    # initialize error metrics
    ace = ACE(bins)
    ece = ECE(bins)
    mce = MCE(bins)

    predictions = []
    all_ace = [ace.measure(validation_set_sm, validation_set_gt)]
    all_ece = [ece.measure(validation_set_sm, validation_set_gt)]
    all_mce = [mce.measure(validation_set_sm, validation_set_gt)]

    # ------------------------------------------

    # build and save models
    for model in models:
        name, instance = model
        print("Build %s model" % name)
        instance.fit(build_set_sm, build_set_gt)

        if save_models:
            instance.save_model("%s/models/%s.pkl" % (domain, name))

    # ------------------------------------------

    # perform predictions
    for model in models:
        _, instance = model
        prediction = instance.transform(validation_set_sm)
        predictions.append(prediction)

        all_ace.append(ace.measure(prediction, validation_set_gt))
        all_ece.append(ece.measure(prediction, validation_set_gt))
        all_mce.append(mce.measure(prediction, validation_set_gt))

    # ------------------------------------------

    # output formatted ECE
    names = [len(x[0]) for x in models]
    buffer = max(names)

    fill = (buffer - len("Default")) * " "
    print("%s%s ACE: %.5f - ECE: %.5f - MCE: %.5f" % ("Default", fill, all_ace[0], all_ece[0], all_mce[0]))
    for i, model in enumerate(models, start=1):
        name, instance = model
        fill = (buffer - len(name)) * " "
        print("%s%s ACE: %.5f - ECE: %.5f - MCE: %.5f" % (name, fill, all_ace[i], all_ece[i], all_mce[i]))

    # ------------------------------------------

    if diagram == 'diagram':

        diagram = ReliabilityDiagram(bins=bins, title_suffix="default")
        diagram.plot(validation_set_sm, validation_set_gt, filename="test.png")
        for i, prediction in enumerate(predictions):
            diagram = ReliabilityDiagram(bins=bins, title_suffix=models[i][0])
            diagram.plot(prediction, validation_set_gt)

    elif diagram is None:
        pass
    else:
        print("Unknown diagram type \'%s\'" % diagram)
        return -1

    return 0


@global_accepts(list, str, int, bool, str)
def cross_validation_5_2(models: list, datafile: str, bins: int, save_models: bool = False, domain: str = '.') -> int:
    """
    5x2 cross validation on given methods on specified dataset.

    Parameters
    ----------
    models : list
        List of tuples with [('<name>', <instance of CalibrationMethod>), ...].
    datafile : str
        Path to datafile which contains two NumPy arrays with keys 'ground_truth' and 'predictions'.
    bins : int
        Number of bins used by ECE, MCE and ReliabilityDiagram.
    save_models : bool, optional, default: False
        True if instances of calibration methods should be stored.
    domain : str, optional, default: "."
        Domain/directory where to store the results.

    Returns
    -------
    int
        0 on success, -1 otherwise
    """

    network = datafile[datafile.rfind("/")+1:datafile.rfind(".npz")]
    seeds = [60932, 29571058, 127519, 23519410, 74198274]

    if not os.path.exists(datafile):
        print("Dataset \'%s\' does not exist" % datafile)
        return -1

    # read NumPy input files
    try:
        with open(datafile, "rb") as open_file:
            npzfile = np.load(open_file)
            ground_truth = npzfile['ground_truth'].squeeze()
            predictions = npzfile['predictions'].squeeze()
    except KeyError:
        print("Key \'ground_truth\' or \'predictions\' not found in file \'%s\'" % datafile)
        return -1

    if len(predictions.shape) == 2:
        n_classes = predictions.shape[1]
    else:
        n_classes = 2

    # initialize error metrics
    ace = ACE(bins)
    ece = ECE(bins)
    mce = MCE(bins)

    all_accuracy = []

    all_ace = []
    all_ece = []
    all_mce = []

    it = 0
    for i, seed in enumerate(seeds):

        np.random.seed(seed)

        # split data set into build set and validation set
        build_set_gt, validation_set_gt, build_set_sm, validation_set_sm = train_test_split(ground_truth, predictions,
                                                                                            random_state=seed,
                                                                                            test_size=0.5,
                                                                                            stratify=ground_truth)

        for j in range(2):

            calibrated_data = {}

            # 5x2 cross validation - flip build/val set after each iteration
            build_set_gt, validation_set_gt = validation_set_gt, build_set_gt
            build_set_sm, validation_set_sm = validation_set_sm, build_set_sm

            # lists for error metrics for current iteration (it)
            it_all_accuracy = []

            it_all_ace = []
            it_all_ece = []
            it_all_mce = []

            if n_classes > 2:
                labels = np.argmax(validation_set_sm, axis=1)
            else:
                labels = np.where(validation_set_sm > 0.5, np.ones_like(validation_set_gt),
                                  np.zeros_like(validation_set_gt))

            accuracy = np.mean(np.where(labels == validation_set_gt, np.ones_like(labels), np.zeros_like(labels)))
            it_all_accuracy.append(accuracy)

            it_all_ace.append(ace.measure(validation_set_sm, validation_set_gt))
            it_all_ece.append(ece.measure(validation_set_sm, validation_set_gt))
            it_all_mce.append(mce.measure(validation_set_sm, validation_set_gt))

            # ------------------------------------------

            # build and save models
            for model in models:
                name, instance = model
                print("Build %s model" % name)

                instance.fit(build_set_sm, build_set_gt)
                if save_models:
                    instance.save_model("%s/models/%s-%s-%d.pkl" % (domain, network, name, i))

                prediction = instance.transform(validation_set_sm)
                calibrated_data[name] = prediction

                if n_classes > 2:
                    if prediction.ndim == 3:
                        prediction = np.mean(prediction, axis=0)

                    labels = np.argmax(prediction, axis=1)
                else:
                    if prediction.ndim == 2:
                        prediction = np.mean(prediction, axis=0)

                    labels = np.where(prediction > 0.5, np.ones_like(validation_set_gt),
                                      np.zeros_like(validation_set_gt))

                accuracy = np.mean(np.where(labels == validation_set_gt, np.ones_like(labels), np.zeros_like(labels)))
                it_all_accuracy.append(accuracy)

                it_all_ace.append(ace.measure(prediction, validation_set_gt))
                it_all_ece.append(ece.measure(prediction, validation_set_gt))
                it_all_mce.append(mce.measure(prediction, validation_set_gt))

            # append lists of current iterations
            all_accuracy.append(it_all_accuracy)
            all_ace.append(it_all_ace)
            all_ece.append(it_all_ece)
            all_mce.append(it_all_mce)

            filename = "%s/results/%s_%02d.npz" % (domain, network, it)
            with open(filename, "wb") as open_file:
                np.savez_compressed(open_file,
                                    train_gt=build_set_gt, test_gt=validation_set_gt,
                                    train_scores=build_set_sm, test_scores=validation_set_sm,
                                    **calibrated_data)

            it += 1

    # convert to NumPy arrays and reduce mean afterwards
    all_accuracy = np.array(all_accuracy)
    all_ace = np.array(all_ace)
    all_ece = np.array(all_ece)
    all_mce = np.array(all_mce)

    all_accuracy = np.mean(all_accuracy, axis=0)
    all_ace = np.mean(all_ace, axis=0)
    all_ece = np.mean(all_ece, axis=0)
    all_mce = np.mean(all_mce, axis=0)

    names = [len(x[0]) for x in models]
    buffer = max(names)

    # ---------------------------------------------------------
    # output formatted ECE
    fill = (buffer - len("Default")) * " "
    print("%s%s Accuracy: %.5f - ACE: %.5f - ECE: %.5f - MCE: %.5f" % ("Default", fill, all_accuracy[0],
                                                                       all_ace[0], all_ece[0], all_mce[0]))

    # ---------------------------------------------------------
    for i, model in enumerate(models, start=1):
        name, instance = model
        fill = (buffer - len(name)) * " "
        print("%s%s Accuracy: %.5f - ACE: %.5f - ECE: %.5f - MCE: %.5f" % (name, fill, all_accuracy[i],
                                                                           all_ace[i], all_ece[i], all_mce[i]))

    return 0
