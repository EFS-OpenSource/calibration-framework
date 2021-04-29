# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import multiprocessing
import numpy as np
from sklearn.model_selection import train_test_split

from netcal.scaling import BetaCalibration, LogisticCalibration, BetaCalibrationDependent, LogisticCalibrationDependent
from netcal.binning import HistogramBinning


def read_predictions(filename: str) -> tuple:
    """ Read npz-files with predictions inferred by a detection model. """

    with open(filename, "rb") as open_file:
        npz = np.load(open_file, allow_pickle=True)

        all_filenames = npz['filenames']
        all_classes = npz['classes']
        all_gt_classes = npz['gt_classes']

        all_boxes = npz['pred_boxes']
        all_gt_boxes = npz['gt_boxes']

        all_scores = npz['scores']
        all_matched = npz['matched']

    return all_filenames, all_classes, all_gt_classes, all_boxes, all_gt_boxes, all_scores, all_matched



def merge_box_data(box_data):
    """ merge relevant box data to a single numpy arrays """

    _, _, _, all_boxes, _, all_scores, all_matched = box_data

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores)
    all_matched = np.concatenate(all_matched)

    return all_boxes, all_scores, all_matched


def eval_method(iteration: int, method_short: str, num_combinations: int,
                train_combinations: list,
                train_matched: np.ndarray,
                test_combinations: list,
                bins_combination: list) -> dict:
    """
    Eval one single method. For multiprocessing it is necessary to create the instance for the calibration
    method within the new process.

    Parameters
    ----------
    iteration : int
        Number of current iteration used for writing the model files
    method_short : str
        Short description string to create the right calibration method within the new process on multiprocessing.
    num_combinations : int
        Total number of different calibration combinations.
    train_combinations : list
        List with all prebuild combinations used for calibration.
    train_matched : list
        List with all prebuild ground truth annotations for each calibration combination.
    test_combinations : list
        List with all prebuild combinations used for testing.
    bins_combination : list
        List with binning schemes for all combinations.

    Returns
    -------
    dict
        Calibration data for each combination on the current method.
    """

    print("Method %s" % method_short)

    # initialize method based on the identifier
    if method_short == "betacal":
        method = BetaCalibration(detection=True)
    elif method_short == "hist":
        method = HistogramBinning(detection=True)
    elif method_short == "lr":
        method = LogisticCalibration(detection=True)
    elif method_short == "lr_dependent":
        method = LogisticCalibrationDependent(detection=True)
    elif method_short == "betacal_dependent":
        method = BetaCalibrationDependent(momentum=True, detection=True)
    else:
        raise AttributeError("Unknown short description")

    # collect calibrated data of each combination
    calibrated_data = {}
    for j in range(num_combinations):
        print("Combination %d method %s" % (j, method_short))
        train_combination, val_combination = train_combinations[j], test_combinations[j]

        # set Histogram binning w.r.t. current combination
        if isinstance(method, HistogramBinning):
            method = HistogramBinning(bins=bins_combination[j], detection=True)

        # fit and save model
        method.fit(train_combination, train_matched)
        method.save_model("models/%s_%s_%d_%02d.pkl" % (network, method_short, j, iteration))

        # perform calibration and save into dict
        calibrated = method.transform(val_combination)
        calibrated_data["%s_c%d" % (method_short, j)] = calibrated
        print("Finished combination %d method %s" % (j, method_short))

    return calibrated_data


def examine_calibration_combinations(all_matched: np.ndarray, all_boxes: np.ndarray, all_scores: np.ndarray):
    """ Core examination routine for our calibration methods for object detection. """

    # fixed seeds to reproduce the results
    seeds = [63091863, 61530583, 213073, 3588059, 38316496, 34393458,
            47543951, 43979170, 37214807, 19239273, 87609388, 91821428,
            17692642, 57440842, 59832019, 77128578, 21112041, 15409117,
            85210406, 14992824]

    num_iterations = 20
    train_split = 0.7

    histogram_bins_0d = 15
    histogram_bins_2d = 5
    histogram_bins_4d = 3

    bins_combination = [histogram_bins_0d, histogram_bins_2d, histogram_bins_2d, histogram_bins_4d]

    # identifier of the different methods
    methods = ['hist', 'betacal', 'lr', 'lr_dependent', 'betacal_dependent']
    num_combinations = 4

    for iteration in range(num_iterations):

        # set seed for current iteration
        print("Iteration %d" % (iteration))
        np.random.seed(seeds[iteration])

        rel_height = all_boxes[:, 2] - all_boxes[:, 0]
        rel_width = all_boxes[:, 3] - all_boxes[:, 1]
        center_x = rel_width * 0.5 + all_boxes[:, 1]
        center_y = rel_height * 0.5 + all_boxes[:, 0]

        train_matched, test_matched, \
        train_scores, test_scores, \
        train_center_y, test_center_y, \
        train_center_x, test_center_x, \
        train_rel_height, test_rel_height, \
        train_rel_width, test_rel_width = train_test_split(all_matched, all_scores, center_y,
                                                          center_x, rel_height, rel_width,
                                                          train_size=train_split, shuffle=True,
                                                          stratify=all_matched,
                                                          random_state=seeds[iteration])
        # --------------------------------------------
        # build all combinations of box properties
        train_combinations = [np.array(train_scores).reshape(-1, 1),
                              np.stack([train_scores, train_center_y, train_center_x], axis=1),
                              np.stack([train_scores, train_rel_height, train_rel_width], axis=1),
                              np.stack(
                                  [train_scores, train_center_y, train_center_x, train_rel_height, train_rel_width],
                                  axis=1
                              )]

        test_combinations = [np.array(test_scores).reshape(-1, 1),
                             np.stack([test_scores, test_center_y, test_center_x], axis=1),
                             np.stack([test_scores, test_rel_height, test_rel_width], axis=1),
                             np.stack([test_scores, test_center_y, test_center_x, test_rel_height, test_rel_width],
                                      axis=1
                                      )
                             ]

        # ------------------------------------------------------------------
        # build calibration algorithms for each possible combination
        # iterate over methods with multiprocessing
        calibrated_data = {}
        with multiprocessing.Pool(processes=len(methods)) as pool:

            # pack arguments of function call
            args = zip([iteration,] * len(methods),
                       methods,
                       [num_combinations,] * len(methods),
                       [train_combinations,] * len(methods),
                       [train_matched,] * len(methods),
                       [test_combinations,] * len(methods),
                       [bins_combination,] * len(methods))

            # multiprocessing function call
            calibrated_data_list = pool.starmap(eval_method, args)
            for entry in calibrated_data_list:
                calibrated_data.update(entry)

        print("Save data iteration %d" % iteration)
        filename = "results/%s_%02d.npz" % (network, iteration)
        with open(filename, "wb") as open_file:
            np.savez_compressed(open_file,
                                train_matched=train_matched, test_matched=test_matched,
                                train_scores=train_scores, test_scores=test_scores,
                                train_center_y=train_center_y, test_center_y=test_center_y,
                                train_center_x=train_center_x, test_center_x=test_center_x,
                                train_rel_height=train_rel_height, test_rel_height=test_rel_height,
                                train_rel_width=train_rel_width, test_rel_width=test_rel_width,
                                **calibrated_data)
    return


if __name__ == '__main__':

    network = "faster-rcnn-resnet-50-iou-0.60"
    filename = "records/%s.npz" % network
    
    box_data = read_predictions(filename)
    all_boxes, all_scores, all_matched = merge_box_data(box_data)

    examine_calibration_combinations(all_matched, all_boxes, all_scores)
