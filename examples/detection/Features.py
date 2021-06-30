# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import List, Dict, Tuple
import numpy as np

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.structures import Instances, Boxes
except ImportError:
    raise ImportError("Need detectron2 to evaluate object detection calibration. You can get the latest version at https://github.com/facebookresearch/detectron2")


def get_features(frames: List[Dict], category_id: int, subset: List, ious: List[float], frame_ids: List) -> Tuple[np.ndarray, List[np.ndarray], List]:
    """
    Compute the calibration features for each frame. Return the preprocessed calibration features as a flattened
    NumPy array holding all predictions in a single array.

    Parameters
    ----------
    frames : List[Dict]
        List of dictionaries holding the input data for each image frame.
    category_id : int
        Integer describing the current category ID for which the features should be extracted
    subset : List[str]
        List with additional features used for calibration. Options are:
        - 'cx'
        - 'cy'
        - 'w'
        - 'h'
    ious : List[float]
        List with IoU scores used for evaluation.
    frame_ids : List
        List with frame indices used for calibration prerpocessing.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray], List]
        Tuple of size 3 containing:
        - NumPy array of shape (N,M) with N predictions over all frames and M features
        - List of NumPy arrays, each of shape (N,) with binary labels (0, 1) indicating a match/not match for each IoU score separately
        - List of length N with image IDs for each prediction. This allows for a reconstruction of the frames later on.
    """

    features, img_ids = [], []
    matched = [[] for _ in ious]

    # iterate over frames
    for frame in frames:

        if frame['image_id'] not in frame_ids:
            continue

        # blank out all predictions that have a different category ID
        categories = frame['category_ids']
        filter = categories == category_id

        if (filter == False).all():
            continue

        # filter out confidences, bounding boxes and matched information
        confidence = frame['scores'][filter]
        bboxes = frame['bboxes'][filter, :] # boxes are in XYWH_ABS
        for i, m in enumerate(frame['matched']):
            matched[i].append(m[filter])

        # convert XYXY_ABS boxes to relative positioning
        img_height, img_width = frame['height'], frame['width']
        rel_cx = (bboxes[:, 0] + 0.5 * bboxes[:, 2]) / float(img_width)
        rel_cy = (bboxes[:, 1] + 0.5 * bboxes[:, 3]) / float(img_height)
        rel_width = bboxes[:, 2] / float(img_width)
        rel_height = bboxes[:, 3] / float(img_height)

        # depending on the specified subset, append different features to the final vector
        input = [confidence]
        if 'cx' in subset:
            input.append(rel_cx)
        if 'cy' in subset:
            input.append(rel_cy)
        if 'w' in subset:
            input.append(rel_width)
        if 'h' in subset:
            input.append(rel_height)

        # also record the respective image ids for each detection
        img_ids.extend([frame['image_id'], ] * confidence.size)
        features.append(np.stack(input, axis=1))

    # in some cases it can occur that no predictions are found for a certain class. In this case, return empty lists
    if len(features) == 0:
        n_features = len(subset) + 1
        matched = [np.empty(0, dtype=np.int32) for _ in ious]
        return np.empty((0, n_features), dtype=np.float32), matched, []

    features = np.concatenate(features, axis=0)
    matched = [np.concatenate(m) for m in matched]
    np.clip(features, np.finfo(np.float32).eps, 1.-np.finfo(np.float32).eps, out=features)

    return features, matched, img_ids
