# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import List, Dict
import os

import numpy as np
from examples.detection.Helper import read_json, read_image_ids, match_frames_with_groundtruth
from examples.detection.Features import get_features

from netcal.binning import HistogramBinning
from netcal.scaling import LogisticCalibration, BetaCalibration
from netcal.scaling import LogisticCalibrationDependent, BetaCalibrationDependent

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
except ImportError:
    raise ImportError("Need detectron2 to evaluate object detection calibration. You can get the latest version at https://github.com/facebookresearch/detectron2")


def calibrate(frames: List[Dict], dataset: str, network: str, subset: List[str], ious: List[float], train_ids: List):
    """
    Perform calibration of the given frames (as list of dicts) for a dedicated dataset with dedicated train_ids.
    The trained models are stored at "calibration/<network>/models/".

    Parameters
    ----------
    frames : List[Dict]
        List of dictionaries holding the input data for each image frame.
    dataset : str
        String of the used dataset (see detectron2 registered datasets).
    network : str
        String describing the base neural network.
    subset : List[str]
        List with additional features used for calibration. Options are:
        - 'cx'
        - 'cy'
        - 'w'
        - 'h'
    ious : List[float]
        List with IoU scores used for evaluation.
    train_ids : List
        List of data frame ids used for calibration training.
    """

    meta = MetadataCatalog.get(dataset)
    model_dir = os.path.join("calibration", network, "models")
    os.makedirs(model_dir, exist_ok=True)

    # reverse mapping of category ids to network class ids (e.g. for COCO dataset)
    if hasattr(meta, "thing_dataset_id_to_contiguous_id"):
        reverse_dictionary = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    else:
        reverse_dictionary = None

    # iterate over classes and perform class-wise calibration
    for i, classname in enumerate(meta.thing_classes):
        category_id = reverse_dictionary[i] if reverse_dictionary is not None else i
        features, matched, _ = get_features(frames, category_id, subset, ious, train_ids)

        if features.size == 0:
            print("No samples for category %s found" % classname)
            continue

        # different binning schemes for different feature dimensions
        if features.shape[1] == 1:
            bins = 15
        elif features.shape[1] == 3:
            bins = 5
        elif features.shape[1] == 5:
            bins = 3
        else:
            raise ValueError("Unknown dimension: %d" % features.shape[1])

        # iterate over IoUs and perform class-wise calibration for each IoU separately
        print("Training: category %d: %d samples" % (category_id, features.shape[0]))
        for iou, m in zip(ious, matched):

            # initialize calibration methods
            histogram = HistogramBinning(bins=bins, detection=True)
            lr = LogisticCalibration(detection=True)
            lr_dependent = LogisticCalibrationDependent()
            betacal = BetaCalibration(detection=True)
            betacal_dependent = BetaCalibrationDependent(momentum_epochs=500)

            # if only negative (or positive) examples are given, calibration is not applicable
            unique = np.unique(m)
            print("Different labels:", unique)
            if len(unique) != 2:
                print("Calibration failed for cls %d as there are only negative samples" % i)
                continue

            # fit and save calibration models
            print("Fit and save histogram binning")
            histogram.fit(features, m)
            histogram.save_model(os.path.join(model_dir, "histogram_%s_iou%.2f_cls-%02d.pkl" % (''.join(subset), iou, i)))

            print("Fit independent logistic calibration")
            lr.fit(features, m)
            lr.save_model(os.path.join(model_dir, "lr_%s_iou%.2f_cls-%02d.pkl" % (''.join(subset), iou, i)))

            print("Fit dependent logistic calibration")
            lr_dependent.fit(features, m)
            lr_dependent.save_model(os.path.join(model_dir, "lr_dependent_%s_iou%.2f_cls-%02d.pkl" % (''.join(subset), iou, i)))

            print("Fit independent beta calibration")
            betacal.fit(features, m)
            betacal.save_model(os.path.join(model_dir, "betacal_%s_iou%.2f_cls-%02d.pkl" % (''.join(subset), iou, i)))

            print("Fit dependent beta calibration")
            betacal_dependent.fit(features, m)
            betacal_dependent.save_model(os.path.join(model_dir, "betacal_dependent_%s_iou%.2f_cls-%02d.pkl" % (''.join(subset), iou, i)))


if __name__ == '__main__':

    # to use this script, perform inference of neural network using Detectron2 first. The predictions are stored as
    # a JSON file in COCO annotations format. This JSON file is understood by the methods in this script.

    # COCO data - Faster RCNN
    filename = "./data/faster-rcnn-coco/inference/coco_instances_results.json"
    network = "COCO-faster-rcnn-threshold-%.f"

    # COCO data - RetinaNet
    # filename = "./data/retinanet-coco/inference/coco_instances_results.json"
    # network = "COCO-retinanet-threshold-%.f"

    dataset = "coco_2017_val"
    train_ids, val_ids = read_image_ids("image_ids_coco.json")

    # Cityscapes data
    # filename = "./data/mask-rcnn-cityscapes/inference/coco_instances_results.json"
    # dataset = "cityscapes_fine_instance_seg_val"
    # network = "Cityscapes-mask-rcnn-threshold-%.f"
    # train_ids, val_ids = read_image_ids("image_ids_cityscapes.json")

    # score threshold used for calibration
    score_threshold = 0.3
    # score_threshold = 0.05

    network = network % score_threshold

    # define different subsets and IoUs that are used for evaluation
    subsets = [[], ['cx', 'cy'], ['w', 'h'], ['cx', 'cy', 'w', 'h']]
    ious = [0.5, 0.75]

    print("Load dataset")
    DatasetCatalog.get(dataset)

    # read frames and match the frames with the according ground-truth boxes. This is mandatory to assess the precision
    frames = read_json(filename, score_threshold=score_threshold)
    frames = match_frames_with_groundtruth(frames, dataset, ious)

    # perform confidence calibration for each subset
    for subset in subsets:
        calibrate(frames, dataset, network, subset, ious, train_ids)
