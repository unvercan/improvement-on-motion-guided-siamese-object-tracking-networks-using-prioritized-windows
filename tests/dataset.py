from __future__ import absolute_import, print_function, unicode_literals

import glob
import io
import json
import os
import shutil
import time
import zipfile
from itertools import chain

import matplotlib
import matplotlib.colors as map_colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import six
import wget
from PIL import Image
from shapely.geometry import box, Polygon

figure_dict = {}
patch_dict = {}


def center_error(rect_1, rect_2):
    center_1 = rect_1[..., :2] + (rect_1[..., 2:] - 1) / 2
    center_2 = rect_2[..., :2] + (rect_2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(center_1 - center_2, 2), axis=-1))
    return errors


def normalized_center_error(rect_1, rect_2):
    center_1 = rect_1[..., :2] + (rect_1[..., 2:] - 1) / 2
    center_2 = rect_2[..., :2] + (rect_2[..., 2:] - 1) / 2
    errors = np.sqrt(
        np.sum(np.power((center_1 - center_2) / np.maximum(np.array([[1., 1.]]), rect_2[:, 2:]), 2), axis=-1))
    return errors


def rect_iou(rect_1, rect_2, bound=None):
    assert rect_1.shape == rect_2.shape
    if bound is not None:
        # bounded rect_1
        rect_1[:, 0] = np.clip(rect_1[:, 0], 0, bound[0])
        rect_1[:, 1] = np.clip(rect_1[:, 1], 0, bound[1])
        rect_1[:, 2] = np.clip(rect_1[:, 2], 0, bound[0] - rect_1[:, 0])
        rect_1[:, 3] = np.clip(rect_1[:, 3], 0, bound[1] - rect_1[:, 1])
        # bounded rect_2
        rect_2[:, 0] = np.clip(rect_2[:, 0], 0, bound[0])
        rect_2[:, 1] = np.clip(rect_2[:, 1], 0, bound[1])
        rect_2[:, 2] = np.clip(rect_2[:, 2], 0, bound[0] - rect_2[:, 0])
        rect_2[:, 3] = np.clip(rect_2[:, 3], 0, bound[1] - rect_2[:, 1])

    rect_inter = intersection(rect_1, rect_2)
    area_inter = np.prod(rect_inter[..., 2:], axis=-1)
    area_1 = np.prod(rect_1[..., 2:], axis=-1)
    area_2 = np.prod(rect_2[..., 2:], axis=-1)
    area_union = area_1 + area_2 - area_inter
    eps = np.finfo(float).eps
    iou = area_inter / (area_union + eps)
    iou = np.clip(iou, 0.0, 1.0)
    return iou


def intersection(rect_1, rect_2):
    assert rect_1.shape == rect_2.shape
    x1 = np.maximum(rect_1[..., 0], rect_2[..., 0])
    y1 = np.maximum(rect_1[..., 1], rect_2[..., 1])
    x2 = np.minimum(rect_1[..., 0] + rect_1[..., 2], rect_2[..., 0] + rect_2[..., 2])
    y2 = np.minimum(rect_1[..., 1] + rect_1[..., 3], rect_2[..., 1] + rect_2[..., 3])
    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)
    return np.stack([x1, y1, w, h]).T


def poly_iou(poly_1, poly_2, bound=None):
    assert poly_1.ndim in [1, 2]
    if poly_1.ndim == 1:
        poly_1 = np.array([poly_1])
        poly_2 = np.array([poly_2])
    assert len(poly_1) == len(poly_2)

    poly_1 = to_polygon(poly_1)
    poly_2 = to_polygon(poly_2)
    if bound is not None:
        bound = box(0, 0, bound[0], bound[1])
        poly_1 = [p.intersection(bound) for p in poly_1]
        poly_2 = [p.intersection(bound) for p in poly_2]

    eps = np.finfo(float).eps
    iou = []
    for p1, p2 in zip(poly_1, poly_2):
        area_inter = p1.intersection(p2).area
        area_union = p1.union(p2).area
        iou.append(area_inter / (area_union + eps))
    iou = np.clip(iou, 0.0, 1.0)
    return iou


def to_polygon_helper(x):
    assert len(x) in [4, 8]
    if len(x) == 4:
        return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
    elif len(x) == 8:
        return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])


def to_polygon(polys):
    if polys.ndim == 1:
        return to_polygon_helper(polys)
    else:
        return [to_polygon_helper(t) for t in polys]


def download(url, filename):
    return wget.download(url, out=filename)


def extract(filename, extract_dir):
    if os.path.splitext(filename)[1] == '.zip':
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir)
        with zipfile.ZipFile(filename) as z:
            z.extractall(extract_dir)
    else:
        raise Exception('Unsupport extension {} of the compressed file {}.'.format(
            os.path.splitext(filename)[1], filename))


def compress(directory_name, save_file):
    shutil.make_archive(save_file, 'zip', directory_name)


def show_frame(image, boxes=None, figure_n=1, pause=0.001, line_width=3, color_map=None, colors=None, legends=None):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1])

    if figure_n not in figure_dict or figure_dict[figure_n].get_size() != image.size[::-1]:
        fig = plt.figure(figure_n)
        plt.axis('off')
        fig.tight_layout()
        figure_dict[figure_n] = plt.imshow(image, cmap=color_map)
    else:
        figure_dict[figure_n].set_data(image)

    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]

        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y'] + list(map_colors.CSS4_COLORS.keys())
        elif isinstance(colors, str):
            colors = [colors]

        if figure_n not in patch_dict:
            patch_dict[figure_n] = []
            for i, box in enumerate(boxes):
                patch_dict[figure_n].append(
                    patches.Rectangle(
                        (box[0], box[1]), box[2], box[3], linewidth=line_width,
                        edgecolor=colors[i % len(colors)], facecolor='none',
                        alpha=0.7 if len(boxes) > 1 else 1.0
                    )
                )
            for patch in patch_dict[figure_n]:
                figure_dict[figure_n].axes.add_patch(patch)
        else:
            for patch, box in zip(patch_dict[figure_n], boxes):
                patch.set_xy((box[0], box[1]))
                patch.set_width(box[2])
                patch.set_height(box[3])

        if legends is not None:
            figure_dict[figure_n].axes.legend(
                patch_dict[figure_n], legends, loc=1, prop={'size': 8}, fancybox=True, framealpha=0.5)

    plt.pause(pause)
    plt.draw()


class Tracker(object):
    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[f, :])

        return boxes, times


class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker', is_deterministic=True)

    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box


def filter_files_otb(filenames):
    filtered_files = []
    for filename in filenames:
        with open(filename, 'r') as f:
            if f.read().strip() == '':
                print('Warning: %s is empty.' % filename)
            else:
                filtered_files.append(filename)

    return filtered_files


def rename_sequences_otb(seq_names):
    # in case some sequences may have multiple targets
    renamed_seqs = []
    for i, seq_name in enumerate(seq_names):
        if seq_names.count(seq_name) == 1:
            renamed_seqs.append(seq_name)
        else:
            ind = seq_names[:i + 1].count(seq_name)
            renamed_seqs.append('%s.%d' % (seq_name, ind))

    return renamed_seqs


class OTB(object):
    otb2013_sequences = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark',
                         'CarScale', 'Coke', 'Couple', 'Crossing', 'David',
                         'David2', 'David3', 'Deer', 'Dog1', 'Doll', 'Dudek',
                         'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace',
                         'Football', 'Football1', 'Freeman1', 'Freeman3',
                         'Freeman4', 'Girl', 'Ironman', 'Jogging', 'Jumping',
                         'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling',
                         'MountainBike', 'Shaking', 'Singer1', 'Singer2',
                         'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv',
                         'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking',
                         'Walking2', 'Woman']

    otb50_sequences = ['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2',
                       'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1', 'Car4',
                       'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds',
                       'David', 'Deer', 'Diving', 'DragonBaby', 'Dudek',
                       'Football', 'Freeman4', 'Girl', 'Human3', 'Human4',
                       'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping',
                       'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam',
                       'Shaking', 'Singer2', 'Skating1', 'Skating2', 'Skiing',
                       'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis',
                       'Walking', 'Walking2', 'Woman']

    otb100_sequences = ['Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board',
                        'Bolt2', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon',
                        'Crossing', 'Dancer', 'Dancer2', 'David2', 'David3',
                        'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2', 'Fish',
                        'FleetFace', 'Football1', 'Freeman1', 'Freeman3',
                        'Girl2', 'Gym', 'Human2', 'Human5', 'Human7', 'Human8',
                        'Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang',
                        'MountainBike', 'Rubik', 'Singer1', 'Skater',
                        'Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans',
                        'Twinnings', 'Vase'] + otb50_sequences

    otb2015_sequences = otb100_sequences

    otb_versions = {
        2013: otb2013_sequences,
        2015: otb2015_sequences,
        'otb2013': otb2013_sequences,
        'otb2015': otb2015_sequences,
        'tb50': otb50_sequences,
        'tb100': otb100_sequences,
        'iztech15otb': ['Trellis', 'Jumping', 'Girl', 'Girl2',
                        'Human7', 'Human3', 'BlurCar3', 'DragonBaby',
                        'Sylvester', 'BlurOwl', 'CarScale', 'CarDark',
                        'Basketball', 'Human4', 'Singer1']
    }

    def __init__(self, root_dir, version='iztech15otb', download=False):
        super(OTB, self).__init__()
        assert version in self.otb_versions

        self.root_dir = root_dir
        self.version = version
        if download:
            self.download(root_dir, version)
        # self._check_integrity(root_dir, version)

        valid_seqs = self.otb_versions[version]
        self.anno_files = sorted(list(chain.from_iterable(glob.glob(
            os.path.join(root_dir, s, 'groundtruth*.txt')) for s in valid_seqs)))
        # remove empty annotation files
        # (e.g., groundtruth_rect.1.txt of Human4)
        self.anno_files = filter_files_otb(self.anno_files)
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        # rename repeated sequence names
        # (e.g., Jogging and Skating2)
        self.seq_names = rename_sequences_otb(self.seq_names)

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if index not in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'img/*.jpg')))

        # special sequences
        # (visit http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html for detail)
        seq_name = self.seq_names[index]
        if seq_name.lower() == 'david':
            img_files = img_files[300 - 1:770]
        elif seq_name.lower() == 'football1':
            img_files = img_files[:74]
        elif seq_name.lower() == 'freeman3':
            img_files = img_files[:460]
        elif seq_name.lower() == 'freeman4':
            img_files = img_files[:283]
        elif seq_name.lower() == 'diving':
            img_files = img_files[:215]

        # to deal with different delimiters
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def download(self, root_dir, version):
        assert version in self.otb_versions
        seq_names = self.otb_versions[version]

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        elif all([os.path.isdir(os.path.join(root_dir, s)) for s in seq_names]):
            print('Files already downloaded.')
            return

        url_fmt = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/%s.zip'
        for seq_name in seq_names:
            seq_dir = os.path.join(root_dir, seq_name)
            if os.path.isdir(seq_dir):
                continue
            url = url_fmt % seq_name
            zip_file = os.path.join(root_dir, seq_name + '.zip')
            print('Downloading to %s...' % zip_file)
            download(url, zip_file)
            print('\nExtracting to %s...' % root_dir)
            extract(zip_file, root_dir)

        return root_dir

    def check_integrity(self, root_dir, version):
        assert version in self.otb_versions
        seq_names = self.otb_versions[version]

        if os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')


def calculate_metrics(boxes, anno):
    # can be modified by children classes
    ious = rect_iou(boxes, anno)
    center_errors = center_error(boxes, anno)
    return ious, center_errors


def _record(record_file, boxes, times):
    # record bounding boxes
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    while not os.path.exists(record_file):
        print('warning: recording failed, retrying...')
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    print('  Results recorded at', record_file)

    # record running times
    time_dir = os.path.join(record_dir, 'times')
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)
    time_file = os.path.join(time_dir, os.path.basename(
        record_file).replace('.txt', '_time.txt'))
    np.savetxt(time_file, times, fmt='%.8f')


class ExperimentOTB(object):
    def __init__(self, root_dir, version='iztech15otb', result_dir='results', report_dir='reports', download=False):
        super(ExperimentOTB, self).__init__()
        self.dataset = OTB(root_dir, version, download=download)
        self.result_dir = os.path.join(result_dir, 'OTB' + str(version))
        self.report_dir = os.path.join(report_dir, 'OTB' + str(version))
        self.threshold_iou = 21
        self.threshold_center = 51

    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(self.result_dir, tracker.name, '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            boxes, times = tracker.track(img_files, anno[0, :], visualize=visualize)
            assert len(boxes) == len(anno)

            # record results
            _record(record_file, boxes, times)

    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)
            success_curve = np.zeros((seq_num, self.threshold_iou))
            precision_curve = np.zeros((seq_num, self.threshold_center))
            speeds = np.zeros(seq_num)

            performance.update({name: {'overall': {}, 'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(self.result_dir, name, '%s.txt' % seq_name)
                boxes = np.loadtxt(record_file, delimiter=',')
                boxes[0] = anno[0]
                if not (len(boxes) == len(anno)):
                    print('warning: %s anno do not match boxes' % seq_name)
                    len_min = min(len(boxes), len(anno))
                    boxes = boxes[:len_min]
                    anno = anno[:len_min]
                assert len(boxes) == len(anno)

                ious, center_errors = calculate_metrics(boxes, anno)
                success_curve[s], precision_curve[s] = self._calc_curves(ious, center_errors)

                # calculate average tracking speed
                time_file = os.path.join(self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': success_curve[s].tolist(),
                    'precision_curve': precision_curve[s].tolist(),
                    'success_score': np.mean(success_curve[s]),
                    'precision_score': precision_curve[s][20],
                    'success_rate': success_curve[s][self.threshold_iou // 2],
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            success_curve = np.mean(success_curve, axis=0)
            precision_curve = np.mean(precision_curve, axis=0)
            success_score = np.mean(success_curve)
            precision_score = precision_curve[20]
            success_rate = success_curve[self.threshold_iou // 2]
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': success_curve.tolist(),
                'precision_curve': precision_curve.tolist(),
                'success_score': success_score,
                'precision_score': precision_score,
                'success_rate': success_rate,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        if plot_curves:
            self.plot_curves(tracker_names)

        return performance

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink'])

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        threshold_iou = np.linspace(0, 1, self.threshold_iou)[np.newaxis, :]
        threshold_center = np.arange(0, self.threshold_center)[np.newaxis, :]

        bin_iou = np.greater(ious, threshold_iou)
        bin_ce = np.less_equal(center_errors, threshold_center)

        success_curve = np.mean(bin_iou, axis=0)
        precision_curve = np.mean(bin_ce, axis=0)

        return success_curve, precision_curve

    def plot_curves(self, tracker_names):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        assert os.path.exists(report_dir), 'No reports found. Run "report" first before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), 'No reports found. Run "report" first before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        success_file = os.path.join(report_dir, 'success_plots.png')
        precision_file = os.path.join(report_dir, 'precision_plots.png')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.threshold_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou, performance[name][key]['success_curve'], markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold', ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1), title='Success plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        print('Saving success plots to', success_file)
        fig.savefig(success_file, bbox_extra_artists=(legend,),
                    bbox_inches='tight', dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.threshold_center)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce, performance[name][key]['precision_curve'], markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold', ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1), title='Precision plots of OPE')
        ax.grid(True)
        fig.tight_layout()
        print('Saving precision plots to', precision_file)
        fig.savefig(precision_file, dpi=300)


def check_integrity_tc128(root_dir):
    seq_names = os.listdir(root_dir)
    seq_names = [n for n in seq_names if not n[0] == '.']

    if os.path.isdir(root_dir) and len(seq_names) > 0:
        # check each sequence folder
        for seq_name in seq_names:
            seq_dir = os.path.join(root_dir, seq_name)
            if not os.path.isdir(seq_dir):
                print('Warning: sequence %s not exists.' % seq_name)
    else:
        # dataset not exists
        raise Exception('Dataset not found or corrupted. You can use download=True to download it.')


def download_tc128(root_dir):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    elif len(os.listdir(root_dir)) > 100:
        print('Files already downloaded.')
        return

    url = 'http://www.dabi.temple.edu/~hbling/data/TColor-128/Temple-color-128.zip'
    zip_file = os.path.join(root_dir, 'Temple-color-128.zip')
    print('Downloading to %s...' % zip_file)
    download(url, zip_file)
    print('\nExtracting to %s...' % root_dir)
    extract(zip_file, root_dir)
    return root_dir


class TColor128(object):
    def __init__(self, root_dir, download=False):
        super(TColor128, self).__init__()
        self.root_dir = root_dir
        if download:
            download_tc128(root_dir)
        check_integrity_tc128(root_dir)

        self.anno_files = sorted(glob.glob(os.path.join(root_dir, '*/*_gt.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        self.range_files = [glob.glob(os.path.join(d, '*_frames.txt'))[0] for d in self.seq_dirs]

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if index not in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        # load valid frame range
        frames = np.loadtxt(self.range_files[index], dtype=int, delimiter=',')
        img_files = [os.path.join(self.seq_dirs[index], 'img/%04d.jpg' % f) for f in range(frames[0], frames[1] + 1)]

        # load annotations
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4
        return img_files, anno

    def __len__(self):
        return len(self.seq_names)


class ExperimentTColor128(ExperimentOTB):
    def __init__(self, root_dir, result_dir='results', report_dir='reports'):
        self.dataset = TColor128(root_dir)
        self.result_dir = os.path.join(result_dir, 'TColor128')
        self.report_dir = os.path.join(report_dir, 'TColor128')
        self.threshold_iou = 21
        self.threshold_center = 51
