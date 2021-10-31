from __future__ import absolute_import

import glob

import numpy as np

from trackers import *

if __name__ == '__main__':
    sequence_name = 'Human3'
    dataset_path = 'C:\\Users\\Unver.Unlu\\Documents\\thesis\\datasets\\otb'
    sequence_path = '{dataset}\\{sequence}\\'.format(
        dataset=dataset_path,
        sequence=sequence_name
    )
    image_paths = sorted(glob.glob(sequence_path + 'img/*.jpg'))
    try:
        annotations = np.loadtxt(sequence_path + 'groundtruth_rect.txt', delimiter=',')
    except:
        annotations = np.loadtxt(sequence_path + 'groundtruth_rect.txt')
    template = annotations[0]
    weight_path = 'C:\\Users\\Unver.Unlu\\PycharmProjects\\master-thesis\\weights\\siamfc_alexnet_e50.pth'
    tracker = SiamFC(weight_path=weight_path)
    tracker.track(
        image_paths=image_paths,
        box=template,
        visualize=True,
        annotations=annotations
    )
