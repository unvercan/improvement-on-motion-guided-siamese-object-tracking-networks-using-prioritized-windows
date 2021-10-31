import numpy as np


def calculate_box_overlap_ratio(box_1, box_2):
    box_1 = np.array(box_1)
    box_2 = np.array(box_2)
    left = np.maximum(box_1[:, 0], box_2[:, 0])
    right = np.minimum(box_1[:, 0] + box_1[:, 2], box_2[:, 0] + box_2[:, 2])
    top = np.maximum(box_1[:, 1], box_2[:, 1])
    bottom = np.minimum(box_1[:, 1] + box_1[:, 3], box_2[:, 1] + box_2[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = box_1[:, 2] * box_1[:, 3] + box_2[:, 2] * box_2[:, 3] - intersection
    iou = intersection / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou


def convert_box_to_center(box):
    box = np.array(box)
    return np.array([(box[:, 0] + (box[:, 2] - 1) / 2), (box[:, 1] + (box[:, 3] - 1) / 2)]).T


def generate_precision_thresholds():
    return np.arange(0, 51, 1)


def generate_success_thresholds():
    return np.arange(0, 1.05, 0.05)


def calculate_precision(groundtruth_center, result_center, thresholds=None):
    if thresholds is None:
        thresholds = generate_precision_thresholds()
    thresholds = np.array(thresholds)
    groundtruth_center = np.array(groundtruth_center)
    result_center = np.array(result_center)
    frame_count = len(groundtruth_center)
    success_error = np.zeros(len(thresholds))
    dist = np.ones(len(groundtruth_center)) * (-1)
    mask = np.sum(groundtruth_center > 0, axis=1) == 2
    dist[mask] = np.sqrt(np.sum(np.power(groundtruth_center[mask] - result_center[mask], 2), axis=1))
    for i in range(len(thresholds)):
        success_error[i] = np.sum(dist <= thresholds[i]) / float(frame_count)
    return success_error


def calculate_success(groundtruth, result, thresholds=None):
    if thresholds is None:
        thresholds = generate_success_thresholds()
    thresholds = np.array(thresholds)
    groundtruth = np.array(groundtruth)
    result = np.array(result)
    frame_count = len(groundtruth)
    success = np.zeros(len(thresholds))
    iou = np.ones(len(groundtruth)) * (-1)
    mask = np.sum(groundtruth > 0, axis=1) == 4
    iou[mask] = calculate_box_overlap_ratio(box_1=groundtruth[mask], box_2=result[mask])
    for i in range(len(thresholds)):
        success[i] = np.sum(iou > thresholds[i]) / float(frame_count)
    return success
