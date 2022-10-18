# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import List, Dict
import os

import numpy as np
import itertools
import csv
from tabulate import tabulate
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt

from netcal.binning import HistogramBinning
from netcal.scaling import LogisticCalibration, BetaCalibration
from netcal.scaling import LogisticCalibrationDependent, BetaCalibrationDependent
from netcal.metrics import ECE, PICP, QCE
from netcal.presentation import ReliabilityDiagram
from examples.detection.Helper import read_json, read_image_ids, match_frames_with_groundtruth
from examples.detection.Features import get_features

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
except ImportError:
    raise ImportError("Need detectron2 to evaluate object detection calibration. You can get the latest version at https://github.com/facebookresearch/detectron2")


def transform(
        frames: List[Dict],
        dataset: str,
        network: str,
        subset: List,
        ious: List,
        test_ids: List[int],
        common_thing_classes: List[str],
        bayesian: bool
):
    """
    After calibration training, evaluate the trained models by several miscalibration metrics. These metrics are:
    D-ECE, Brier, NLL. Also capture area under precision-recall curve (AUPRC).
    All results are stored at "./output/<network>".

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
    test_ids : List
        List of data frame ids used for calibration testing.
    """

    n_stochastic = 1000
    random_seed = 0
    quantiles = np.linspace(0.05, 0.95, 21)

    # get meta information and specify all relevant paths
    meta = MetadataCatalog.get(dataset)
    model_dir = os.path.join("calibration" if not bayesian else "bayesian_calibration", network, "models")
    output_dir = os.path.join("output" if not bayesian else "bayesian_output", network, dataset)
    diagram_path = os.path.join(output_dir, "diagrams", ''.join(subset) if len(subset) > 0 else "confidence")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(diagram_path, exist_ok=True)

    # calibration methods that have also been used for calibration training
    methods = [
        ("lr", LogisticCalibration),
        ("lr_dependent", LogisticCalibrationDependent),
        ("betacal", BetaCalibration),
        ("betacal_dependent", BetaCalibrationDependent)
    ]

    if not bayesian:
        methods.append(("histogram", HistogramBinning))

    # reverse mapping of category ids to network class ids (e.g. for COCO dataset)
    if hasattr(meta, "thing_dataset_id_to_contiguous_id"):
        reverse_dictionary = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    else:
        reverse_dictionary = None

    if common_thing_classes is not None:
        eval_classes = list(filter(lambda x: x[1] in common_thing_classes, enumerate(meta.thing_classes)))
    else:
        eval_classes = enumerate(meta.thing_classes)

    n_classes_in_dataset = len(meta.thing_classes)

    # lists and placeholders for evaluation metrics
    n_samples_total = 0
    n_samples_per_class = np.zeros(n_classes_in_dataset, dtype=np.int64)

    # fill anything by NaN so that e.g. empty classes are not considered for the final computation
    dece_per_class = np.full((len(methods)+1, len(ious), n_classes_in_dataset), fill_value=np.nan, dtype=np.float32)

    brier_per_class = np.full((len(methods)+1, len(ious), n_classes_in_dataset), fill_value=np.nan, dtype=np.float32)
    nll_per_class = np.full((len(methods)+1, len(ious), n_classes_in_dataset), fill_value=np.nan, dtype=np.float32)
    average_precision = np.full((len(methods)+1, len(ious), n_classes_in_dataset), fill_value=np.nan, dtype=np.float32)

    # uncertainty metrics
    picp_per_class = np.full((len(methods)+1, len(ious), n_classes_in_dataset), fill_value=np.nan, dtype=np.float32)
    mpiw_per_class = np.full((len(methods)+1, len(ious), n_classes_in_dataset), fill_value=np.nan, dtype=np.float32)
    qce_per_class = np.full((len(methods)+1, len(ious), n_classes_in_dataset), fill_value=np.nan, dtype=np.float32)

    # -----------------------------------------------------
    # visualization routine
    diagram0d = ReliabilityDiagram(bins=20, detection=True, sample_threshold=8)
    diagram1d = ReliabilityDiagram(bins=[5, 15], detection=True, sample_threshold=3, fmin=0, fmax=0.3)
    diagram2d = ReliabilityDiagram(bins=[6, 9, 9], detection=True, sample_threshold=2, fmin=0, fmax=0.3)

    def plot(f: np.ndarray, m: np.ndarray, title: str, formatter: str):
        # Define function for diagram output

        # plot baseline miscalibration
        figures = [diagram0d.plot(
            f[..., :1], m,
            tikz=False,
            title_suffix=title,
            filename=formatter % "0d",
            uncertainty="mean" if bayesian else None
        )]

        # plot all additional features in 1D miscalibration plots
        for i, fname in enumerate(['cx', 'cy', 'w', 'h']):
            figures.append(
                diagram1d.plot(
                    f[..., (0, i + 1)], m,
                    tikz=False,
                    feature_names=[fname],
                    title_suffix=title,
                    filename=formatter % ("1d_%s" % fname),
                    uncertainty="mean" if bayesian else None
                )
            )

        # finally, plot all feature combinations of size 2
        for (i, fname1), (j, fname2) in itertools.combinations(enumerate(['cx', 'cy', 'w', 'h']), 2):
            figures.append(
                diagram2d.plot(
                    f[..., (0, i + 1, j + 1)], m,
                    tikz=False,
                    feature_names=[fname1, fname2],
                    title_suffix=title,
                    filename=formatter % ("2d_%s_%s" % (fname1, fname2)),
                    uncertainty="mean" if bayesian else None)
            )

        # free memory space
        for fig in figures:
            plt.close(fig)

    # -----------------------------------------------------

    # iterate over all classes that are present in the current dataset
    for i, classname in eval_classes:

        # get calibration features for selected class
        category_id = reverse_dictionary[i] if reverse_dictionary is not None else i
        features, matched, img_ids = get_features(frames, category_id, subset, ious, test_ids)
        all_features, _, _ = get_features(frames, category_id, ['cx', 'cy', 'w', 'h'], ious, test_ids)

        if features.size == 0:
            print("No samples for category %s found" % classname)
            continue

        n_samples, n_features = features.shape

        # different binning schemes for different feature dimensions
        if features.shape[1] == 1:
            bins = 20
        elif features.shape[1] == 3:
            bins = 8
        elif features.shape[1] == 5:
            bins = 5
        else:
            raise ValueError("Unknown dimension: %d" % features.shape[1])

        # define D-ECE metric
        dece = ECE(bins=bins, detection=True, sample_threshold=8)
        picp = PICP(bins=bins, detection=True, sample_threshold=8)
        qce = QCE(bins=bins, detection=True, marginal=True, sample_threshold=8)

        n_samples_per_class[i] = features.shape[0]
        n_samples_total += features.shape[0]

        # failed flag is required to optionally blank failed or non-present classes during evaluation
        # i.e., if a metric returns NaN
        failed = False

        # perform evaluation for each category separately
        print("Inference: category %d (%s): %d samples" % (category_id, classname, features.shape[0]))
        for j, (iou, m) in enumerate(zip(ious, matched)):

            score = average_precision_score(m, features[:, 0])
            if not np.isfinite(score) or np.isnan(score):
                failed = True

            # compute average precision, Brier, NLL and ECE
            else:
                brier_per_class[0, j, i] = np.mean(np.square(features[:, 0] - m))
                nll_per_class[0, j, i] = -np.mean(m * np.log(features[:, 0]) + (1. - m) * np.log(1. - features[:, 0]))
                dece_per_class[0, j, i] = dece.measure(features, m)
                average_precision[0, j, i] = score

            # uncertainty metrics are not defined for uncalibrated baseline
            picp_per_class[0, j, i] = np.nan
            mpiw_per_class[0, j, i] = np.nan
            qce_per_class[0, j, i] = np.nan

            diagramname = os.path.join(diagram_path, "default_cls-%02d_iou%.2f" % (i, iou) + "_%s.png")
            plot(all_features, m, title="default", formatter=diagramname)

            # start calibration evaluation for each method separately
            for k, (name, method) in enumerate(methods, start=1):
                instance = method()

                try:
                    model_path = os.path.join(model_dir, "%s_%s_iou%.2f_cls-%02d.pkl" % (name, ''.join(subset), iou, i))
                    print("Load %s at path: \'%s\'" % (name, model_path))
                    instance.load_model(model_path)

                    if bayesian:
                        calibrated = instance.transform(features, num_samples=n_stochastic, random_state=random_seed)
                        calibrated_mean = np.mean(calibrated, axis=0)
                    else:
                        calibrated = instance.transform(features)
                        calibrated_mean = calibrated

                    # perform clipping
                    np.clip(calibrated, np.finfo(np.float32).eps, 1. - np.finfo(np.float32).eps, out=calibrated)
                    np.clip(calibrated_mean, np.finfo(np.float32).eps, 1. - np.finfo(np.float32).eps, out=calibrated_mean)

                    score = average_precision_score(m, calibrated_mean)
                    if not np.isfinite(score) or np.isnan(score):
                        raise ValueError("Couldn't compute AUPRC score")

                    average_precision[k, j, i] = score

                    brier_per_class[k, j, i] = np.mean(np.square(calibrated_mean - m))
                    nll_per_class[k, j, i] = -np.mean(m * np.log(calibrated_mean) + (1. - m) * np.log(1. - calibrated_mean))

                    input = np.concatenate(
                        (
                            np.expand_dims(calibrated, axis=-1),  # ([t,] n, 1)
                            features[:, 1:] if not bayesian else np.broadcast_to(features[None, :, 1:], (n_stochastic, n_samples, n_features-1))  # ([t,] n, d-1)
                        ),
                        axis=-1
                    )  # ([t,] n, d)
                    dece_per_class[k, j, i] = dece.measure(input, m, uncertainty="mean" if bayesian else None)

                    # now measure quality of bayesian output distributions
                    if bayesian:
                        picp_score, mpiw_score = picp.measure(input, m, uncertainty="mean")
                        picp_per_class[k, j, i] = picp_score
                        mpiw_per_class[k, j, i] = mpiw_score
                        qce_per_class[k, j, i] = qce.measure(input, m, quantiles, kind="confidence", reduction="mean")

                    diagramname = os.path.join(diagram_path, "%s_cls-%02d_iou%.2f" % (name, i, iou) + "_%s.png")
                    input = np.concatenate(
                        (
                            np.expand_dims(calibrated, axis=-1),  # ([t,] n, 1)
                            all_features[:, 1:] if not bayesian else np.broadcast_to(all_features[None, :, 1:], (n_stochastic, n_samples, 4))  # ([t,] n, d-1)
                        ),
                        axis=-1
                    )
                    plot(input, m, title=name, formatter=diagramname)

                except (FileNotFoundError, ValueError) as e:
                    print(e)
                    print("Could not find weight file ", os.path.join(model_dir, "%s_%s_iou%.2f_cls-%02d.pkl" % (name, ''.join(subset), iou, i)))
                    print("Disable evaluation for class %d" % i)

                    brier_per_class[k, j, i] = np.nan
                    nll_per_class[k, j, i] = np.nan
                    dece_per_class[k, j, i] = np.nan
                    average_precision[k, j, i] = np.nan

                    picp_per_class[k, j, i] = np.nan
                    mpiw_per_class[k, j, i] = np.nan
                    qce_per_class[k, j, i] = np.nan

                    failed = True

        if failed:
            n_samples_total -= n_samples_per_class[i]
            n_samples_per_class[i] = 0

    # convert all lists to NumPy arrays
    weights = n_samples_per_class / n_samples_total  # (classes,)

    def get_average_unweighted_weighted(array: np.ndarray):
        # compute a feed-forward average and and a weighted counter-part

        unweighted = np.nanmean(array, axis=2)
        current_weights = (~np.isnan(array)) * weights  # (methods, iou, classes)
        current_weights = current_weights / np.sum(current_weights, axis=2, keepdims=True)  # rescale so that the weights sum up to 1
        weighted = np.nansum(array * current_weights, axis=2)

        return unweighted, weighted

    brier_global, weighted_brier_global = get_average_unweighted_weighted(brier_per_class)
    nll_global, weighted_nll_global = get_average_unweighted_weighted(nll_per_class)
    dece_global, weighted_dece_global = get_average_unweighted_weighted(dece_per_class)
    average_precision_macro, average_precision_weighted = get_average_unweighted_weighted(average_precision)
    picp_global, weighted_picp_global = get_average_unweighted_weighted(picp_per_class)
    mpiw_global, weighted_mpiw_global = get_average_unweighted_weighted(mpiw_per_class)
    qce_global, weighted_qce_global = get_average_unweighted_weighted(qce_per_class)

    # use tabulate library to visualize the evaluation results
    header = []
    body = [['default']]
    body.extend([[name] for name, method in methods])
    for i, iou in enumerate(ious):

        header.extend([
            'D-ECE(w) @ IoU %.2f' % iou,
            'D-ECE @ IoU %.2f' % iou,
            'Brier(w) @ IoU %.2f' % iou,
            'Brier @ IoU %.2f' % iou,
            'NLL(w) @ IoU %.2f' % iou,
            'NLL @ IoU %.2f' % iou,
            'AP(w) @ IoU %.2f' % iou,
            'AP @ IoU %.2f' % iou,
            'PICP(w) @ IoU %.2f' % iou,
            'PICP @ IoU %.2f' % iou,
            'MPIW(w) @ IoU %.2f' % iou,
            'MPIW @ IoU %.2f' % iou,
            'QCE(w) @ IoU %.2f' % iou,
            'QCE @ IoU %.2f' % iou,
        ])
        body[0].extend([
            weighted_dece_global[0, i],
            dece_global[0, i],
            weighted_brier_global[0, i],
            brier_global[0, i],
            weighted_nll_global[0, i],
            nll_global[0, i],
            average_precision_weighted[0, i],
            average_precision_macro[0, i],
            weighted_picp_global[0, i],
            picp_global[0, i],
            weighted_mpiw_global[0, i],
            mpiw_global[0, i],
            weighted_qce_global[0, i],
            qce_global[0, i]
        ])
        for k, (name, method) in enumerate(methods):
            body[k+1].extend([
                weighted_dece_global[k+1, i],
                dece_global[k+1, i],
                weighted_brier_global[k+1, i],
                brier_global[k+1, i],
                weighted_nll_global[k+1, i],
                nll_global[k+1, i],
                average_precision_weighted[k+1, i],
                average_precision_macro[k+1, i],
                weighted_picp_global[k+1, i],
                picp_global[k+1, i],
                weighted_mpiw_global[k+1, i],
                mpiw_global[k+1, i],
                weighted_qce_global[k+1, i],
                qce_global[k+1, i]
            ])

    results = [header, *body]

    # also write the evaluation results to CSV format
    print("\nEvaluation Results:")
    print(tabulate(results, headers="firstrow"))
    with open(os.path.join(output_dir, "results_%s.csv" % ''.join(subset)), "w") as open_file:
        writer = csv.writer(open_file)
        writer.writerow(["method", ] + results[0])
        writer.writerows(results[1:])


if __name__ == '__main__':

    # to use this script, perform inference of neural network using Detectron2 first. The predictions are stored as
    # a JSON file in COCO annotations format. This JSON file is understood by the methods in this script.

    # set 'bayesian' to true if you want to use stochastic variational inference for calibration training
    bayesian = False

    # classes that are common within Cityscapes and COCO. Is everything in Cityscapes except class "rider"
    # only required when performing cross-evaluation
    common_thing_classes = ['person', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    # common_thing_classes = None

    # COCO data - Faster RCNN
    # filename = "./data/faster-rcnn-coco/inference/coco_instances_results.json"
    # filename = "./data/faster-rcnn-coco-eval-on-cityscapes/converted_coco_instances_results.json"
    # network = "COCO-faster-rcnn-threshold-%.2f"

    # COCO data - RetinaNet
    # filename = "./data/retinanet-coco/inference/coco_instances_results.json"
    filename = "./data/retinanet-coco-eval-on-cityscapes/converted_coco_instances_results.json"
    network = "COCO-retinanet-threshold-%.2f"

    # Cityscapes data
    # filename = "./data/mask-rcnn-cityscapes/inference/coco_instances_results.json"
    # filename = "./data/mask-rcnn-cityscapes-eval-on-coco/converted_coco_instances_results.json"
    # network = "Cityscapes-mask-rcnn-threshold-%.2f"

    # dataset = "coco_2017_val"
    # train_ids, val_ids = read_image_ids("image_ids_coco.json")

    dataset = "cityscapes_fine_instance_seg_val"
    train_ids, val_ids = read_image_ids("image_ids_cityscapes.json")

    score_threshold = 0.3
    # score_threshold = 0.05

    network = network % score_threshold

    # subsets = [[], ['cx', 'cy'], ['w', 'h'], ['cx', 'cy', 'w', 'h']]
    subsets = [[], ['cx', 'cy', 'w', 'h']]
    ious = [0.5, 0.75]

    # read frames and match the frames with the according ground-truth boxes. This is mandatory to assess the precision
    DatasetCatalog.get(dataset)
    frames = read_json(filename, score_threshold=score_threshold)
    frames = match_frames_with_groundtruth(frames, dataset, ious)

    # perform confidence calibration evaluation for each subset
    for subset in subsets:
        transform(frames, dataset, network, subset, ious, val_ids, common_thing_classes, bayesian)
