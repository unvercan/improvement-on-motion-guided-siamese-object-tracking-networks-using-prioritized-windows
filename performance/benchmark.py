from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from performance.calculation import generate_success_thresholds, generate_precision_thresholds
from performance.common import COLOR, LINE, IZTECH15TC_SEQUENCES, TC128_SEQUENCES, OTB100_SEQUENCES, TRACKER_COLORS, \
    TRACKER_LINES, IZTECH15OTB_SEQUENCES
from performance.result import load_results


def show_results(results=None, trackers=None, datasets=None, sequences=None, attributes=None, title=''):
    result_line_format = '|{tracker:^30}|{dataset:^30}|{sequence:^25}|{attributes:^40}' \
                         '|{success:^15.3f}|{precision:^15.3f}|{fps:^15.3f}|'
    header_line_format = '|{tracker:^30}|{dataset:^30}|{sequence:^25}|{attributes:^40}' \
                         '|{success:^15}|{precision:^15}|{fps:^15}|'
    header_line = header_line_format.format(
        tracker='Tracker',
        dataset='Dataset',
        sequence='Sequence',
        attributes='Attributes',
        success='Success',
        precision='Precision',
        fps='FPS'
    )
    if results is None:
        results = list()
    if trackers is None:
        trackers = list()
    if datasets is None:
        datasets = list()
    if sequences is None:
        sequences = list()
    if attributes is None:
        attributes = list()
    if len(results) > 0:
        if len(title) > 0:
            title_line_format = '|{title:^' + str(len(header_line) - 2) + '}|'
            title_line = title_line_format.format(title=title)
            print('-' * len(header_line))
            print(title_line)
        print('-' * len(header_line))
        print(header_line)
        print('-' * len(header_line))
        if len(results) > 0:
            for result in results:
                if (trackers is None or len(trackers) == 0 or result.tracker in trackers) \
                        and (datasets is None or len(datasets) == 0 or result.dataset in datasets) \
                        and (sequences is None or len(sequences) == 0 or result.sequence in sequences) \
                        and (attributes is None or len(attributes) == 0
                             or any(attribute in result.attributes for attribute in attributes)):
                    result_line = result_line_format.format(
                        tracker=result.tracker,
                        dataset=result.dataset,
                        sequence=result.sequence,
                        attributes=','.join(result.attributes),
                        success=float(str(result.success)[:5]),
                        precision=float(str(result.precision)[:5]),
                        fps=float(str(result.fps)[:5])
                    )
                    print(result_line)
        print('-' * len(header_line))


def show_results_overall(results=None, trackers=None, datasets=None, sequences=None, attributes=None, title=''):
    result_line_format = '|{tracker:^30}|{dataset:^30}|{success:^15.3f}|{precision:^15.3f}|{fps:^15.3f}|'
    header_line_format = '|{tracker:^30}|{dataset:^30}|{success:^15}|{precision:^15}|{fps:^15}|'
    header_line = header_line_format.format(
        tracker='Tracker',
        dataset='Dataset',
        success='Success',
        precision='Precision',
        fps='FPS'
    )
    if results is None:
        results = list()
    if trackers is None:
        trackers = list()
    if datasets is None:
        datasets = list()
    if sequences is None:
        sequences = list()
    if attributes is None:
        attributes = list()
    if len(results) > 0:
        if len(title) > 0:
            title_line_format = '|{title:^' + str(len(header_line) - 2) + '}|'
            title_line = title_line_format.format(title=title)
            print('-' * len(header_line))
            print(title_line)
        print('-' * len(header_line))
        print(header_line)
        print('-' * len(header_line))
        overall = dict()
        for result in results:
            if (trackers is None or len(trackers) == 0 or result.tracker in trackers) \
                    and (datasets is None or len(datasets) == 0 or result.dataset in datasets) \
                    and (sequences is None or len(sequences) == 0 or result.sequence in sequences) \
                    and (attributes is None or len(attributes) == 0
                         or any(attribute in result.attributes for attribute in attributes)):
                if result.tracker not in overall.keys():
                    overall[result.tracker] = dict()
                if result.dataset not in overall[result.tracker].keys():
                    overall[result.tracker][result.dataset] = dict()
                    overall[result.tracker][result.dataset]['success'] = list()
                    overall[result.tracker][result.dataset]['precision'] = list()
                    overall[result.tracker][result.dataset]['fps'] = list()
                overall[result.tracker][result.dataset]['success'].append(result.success)
                overall[result.tracker][result.dataset]['precision'].append(result.precision)
                overall[result.tracker][result.dataset]['fps'].append(result.fps)
        for tracker in overall.keys():
            for dataset in overall[tracker].keys():
                overall[tracker][dataset]['success'] = np.mean(overall[tracker][dataset]['success'])
                overall[tracker][dataset]['precision'] = np.mean(overall[tracker][dataset]['precision'])
                overall[tracker][dataset]['fps'] = np.mean(overall[tracker][dataset]['fps'])
                result_line = result_line_format.format(
                    tracker=tracker,
                    dataset=dataset,
                    success=float(str(overall[tracker][dataset]['success'])[:5]),
                    precision=float(str(overall[tracker][dataset]['precision'])[:5]),
                    fps=float(str(overall[tracker][dataset]['fps'])[:5])
                )
                print(result_line)
        print('-' * len(header_line))


def plot_success(results=None, thresholds=None, trackers=None, datasets=None,
                 sequences=None, attributes=None, title='', dpi=100):
    if thresholds is None:
        thresholds = generate_success_thresholds()
    if results is None:
        results = list()
    if trackers is None:
        trackers = list()
    if datasets is None:
        datasets = list()
    if sequences is None:
        sequences = list()
    if attributes is None:
        attributes = list()
    if len(results) > 0:
        success = dict()
        for result in results:
            if (trackers is None or len(trackers) == 0 or result.tracker in trackers) \
                    and (datasets is None or len(datasets) == 0 or result.dataset in datasets) \
                    and (sequences is None or len(sequences) == 0 or result.sequence in sequences) \
                    and (attributes is None or len(attributes) == 0
                         or any(attribute in result.attributes for attribute in attributes)):
                if result.tracker not in success.keys():
                    success[result.tracker] = list()
                success[result.tracker].append(result.success)
        for tracker in success.keys():
            success[tracker] = np.mean(success[tracker])
        success = OrderedDict(sorted(success.items(), key=lambda x: x[1], reverse=True))
        if len(success) > 0:
            figure, axes = plt.subplots()
            axes.grid(b=True)
            axes.set_aspect(1)
            plt.xlabel('Overlap')
            plt.ylabel('Success')
            if title is None or len(title) == 0:
                title = 'Success plot'
            plt.title(title)
            plt.axis([0, 1, 0, 1])
            for tracker_index, tracker in enumerate(success.keys()):
                if trackers is None or len(trackers) == 0 or tracker in trackers:
                    color = TRACKER_COLORS.get(tracker)
                    if color is None:
                        color = COLOR[tracker_index]
                    line = TRACKER_LINES.get(tracker)
                    if line is None:
                        line = LINE[tracker_index]
                    label = "[{auc:.3f}] {tracker}".format(auc=float(str(success[tracker])[:5]), tracker=tracker)
                    success_tracker = list()
                    for result in results:
                        if result.tracker == tracker \
                                and (datasets is None or len(datasets) == 0 or result.dataset in datasets) \
                                and (sequences is None or len(sequences) == 0 or result.sequence in sequences) \
                                and (attributes is None or len(attributes) == 0
                                     or any(attribute in result.attributes for attribute in attributes)):
                            success_tracker.append(result.successes)
                    success_tracker = np.mean(success_tracker, axis=0)
                    plt.plot(thresholds, success_tracker, color=color, linestyle=line, label=label, linewidth=2)
            axes.legend(loc='lower left', labelspacing=0.2)
            axes.autoscale(enable=True, axis='both', tight=True)
            x_min, x_max, y_min, y_max = plt.axis()
            axes.autoscale(enable=False)
            y_max += 0.03
            y_min = 0
            plt.axis([x_min, x_max, y_min, y_max])
            plt.xticks(np.arange(x_min, x_max + 0.01, 0.1))
            plt.yticks(np.arange(y_min, y_max, 0.1))
            axes.set_aspect((x_max - x_min) / (y_max - y_min))
            plt.savefig(title, dpi=dpi)
            plt.show()


def plot_precision(results=None, thresholds=None, trackers=None, datasets=None,
                   sequences=None, attributes=None, title='', dpi=100):
    if thresholds is None:
        thresholds = generate_precision_thresholds()
    if results is None:
        results = list()
    if trackers is None:
        trackers = list()
    if datasets is None:
        datasets = list()
    if sequences is None:
        sequences = list()
    if attributes is None:
        attributes = list()
    if len(results) > 0:
        precision = dict()
        for result in results:
            if (trackers is None or len(trackers) == 0 or result.tracker in trackers) \
                    and (datasets is None or len(datasets) == 0 or result.dataset in datasets) \
                    and (sequences is None or len(sequences) == 0 or result.sequence in sequences) \
                    and (attributes is None or len(attributes) == 0
                         or any(attribute in result.attributes for attribute in attributes)):
                if result.tracker not in precision.keys():
                    precision[result.tracker] = list()
                precision[result.tracker].append(result.precision)
        for tracker in precision.keys():
            precision[tracker] = np.mean(precision[tracker])
        precision = OrderedDict(sorted(precision.items(), key=lambda x: x[1], reverse=True))
        if len(precision) > 0:
            figure, axes = plt.subplots()
            axes.grid(b=True)
            axes.set_aspect(50)
            plt.xlabel('Center Location Error')
            plt.ylabel('Precision')
            if title is None or len(title) == 0:
                title = 'Precision plot'
            plt.title(title)
            plt.axis([0, 50, 0, 1])
            for tracker_index, tracker in enumerate(precision.keys()):
                if trackers is None or len(trackers) == 0 or tracker in trackers:
                    color = TRACKER_COLORS.get(tracker)
                    if color is None:
                        color = COLOR[tracker_index]
                    line = TRACKER_LINES.get(tracker)
                    if line is None:
                        line = LINE[tracker_index]
                    label = "[{auc:.3f}] {tracker}".format(auc=float(str(precision[tracker])[:5]), tracker=tracker)
                    precision_tracker = list()
                    for result in results:
                        if result.tracker == tracker \
                                and (datasets is None or len(datasets) == 0 or result.dataset in datasets) \
                                and (sequences is None or len(sequences) == 0 or result.sequence in sequences) \
                                and (attributes is None or len(attributes) == 0
                                     or any(attribute in result.attributes for attribute in attributes)):
                            precision_tracker.append(result.precisions)
                    precision_tracker = np.mean(precision_tracker, axis=0)
                    plt.plot(thresholds, precision_tracker, color=color, linestyle=line, label=label, linewidth=2)
            axes.legend(loc='lower right', labelspacing=0.2)
            axes.autoscale(enable=True, axis='both', tight=True)
            x_min, x_max, y_min, y_max = plt.axis()
            axes.autoscale(enable=False)
            y_max += 0.03
            y_min = 0
            plt.axis([x_min, x_max, y_min, y_max])
            plt.xticks(np.arange(x_min, x_max + 0.01, 5))
            plt.yticks(np.arange(y_min, y_max, 0.1))
            axes.set_aspect((x_max - x_min) / (y_max - y_min))
            plt.savefig(title, dpi=dpi)
            plt.show()


# Results
results_path = 'C:\\Users\\Unver.Unlu\\PycharmProjects\\master-thesis\\performance\\results.json'
results_combined = load_results(file_path=results_path)
results_combined = sorted(results_combined, key=lambda result: (result.dataset, result.tracker, result.sequence))

# OTB100
results_OTB100 = list()
for result in results_combined:
    sequence = result.sequence
    if len(result.sequence.split('.')) > 1:
        sequence = ''.join(result.sequence.split('.')[:-1])
    if (sequence in OTB100_SEQUENCES) and (result not in results_OTB100) and (result.dataset == 'OTB100'):
        results_OTB100.append(result)

results_OTB100 = sorted(results_OTB100, key=lambda result: (result.tracker, result.sequence))

"""
k = list()
for otb100seq in OTB100_SEQUENCES:
    a = list(filter(lambda r: r.sequence == otb100seq, results_OTB100))
    adaptive = list(filter(lambda r: r.tracker == 'AdaptiveKalmanSiam', a))[0]
    kalman = list(filter(lambda r: r.tracker == 'KalmanSiam', a))[0]
    siam = list(filter(lambda r: r.tracker == 'SiamFC', a))[0]

    difference1 = (adaptive.success - kalman.success)
    difference2 = (adaptive.success - siam.success)

    k.append({
        'diff1': difference1,
        'diff2': difference2,
        'seq': otb100seq,
        'success': adaptive.success,
        'precision': adaptive.precision
    })

k = sorted(k, key=lambda j: j['diff1'] ** 5 + j['diff2'] ** 5, reverse=True)

for o in k:
    if o['success'] > 0.4 and o['precision'] > 0.4:
        print('{sequence}, {success:.3f}, {precision:.3f}, {difference1:.3f}, {difference2:.3f}'.format(
            sequence=o['seq'], success=o['success'], precision=o['precision'],
            difference1=o['diff1'], difference2=o['diff2']))
"""

OTB100_by_attribute = dict()
for result in results_OTB100:
    for attribute in result.attributes:
        if attribute not in OTB100_by_attribute.keys():
            OTB100_by_attribute[attribute] = list()
        OTB100_by_attribute[attribute].append(result)

OTB100_by_attribute = OrderedDict(sorted(OTB100_by_attribute.items(), key=lambda item: item[0]))

for attribute in OTB100_by_attribute.keys():
    OTB100_by_attribute[attribute] = sorted(OTB100_by_attribute[attribute],
                                            key=lambda result: (result.tracker, result.sequence))

for attribute in OTB100_by_attribute.keys():
    plot_success(results=OTB100_by_attribute[attribute],
                 title='Success Plot on {dataset} for {attribute} using OPE'
                 .format(dataset='OTB100', attribute=attribute))
    plot_precision(results=OTB100_by_attribute[attribute],
                   title='Precision Plot on {dataset} for {attribute} using OPE'
                   .format(dataset='OTB100', attribute=attribute))
    show_results(results=OTB100_by_attribute[attribute],
                 title='Results on {dataset} for {attribute} using OPE'
                 .format(dataset='OTB100', attribute=attribute))

plot_success(results=results_OTB100,
             title='Success Plot on {dataset} using OPE'
             .format(dataset='OTB100'))
plot_precision(results=results_OTB100,
               title='Precision Plot on {dataset} using OPE'
               .format(dataset='OTB100'))
show_results(results=results_OTB100,
             title='Results on {dataset} using OPE'
             .format(dataset='OTB100'))

# TC128
results_TC128 = list()
for result in results_combined:
    if (result.sequence in TC128_SEQUENCES) and (result not in results_TC128) and (result.dataset == 'TC128'):
        results_TC128.append(result)

results_TC128 = sorted(results_TC128, key=lambda result: (result.tracker, result.sequence))

"""
k = list()
for seq in TC128_SEQUENCES:
    a = list(filter(lambda r: r.sequence == seq, results_TC128))
    adaptive = list(filter(lambda r: r.tracker == 'AdaptiveKalmanSiam', a))[0]
    kalman = list(filter(lambda r: r.tracker == 'KalmanSiam', a))[0]
    siam = list(filter(lambda r: r.tracker == 'SiamFC', a))[0]

    difference1 = (adaptive.success - kalman.success)
    difference2 = (adaptive.success - siam.success)

    k.append({
        'diff1': difference1,
        'diff2': difference2,
        'seq': seq,
        'success': adaptive.success,
        'precision': adaptive.precision
    })

k = sorted(k, key=lambda j: j['diff1'] ** 5 + j['diff2'] ** 5, reverse=True)

for o in k:
    if o['success'] > 0.4 and o['precision'] > 0.4:
        print('{sequence}, {success:.3f}, {precision:.3f}, {difference1:.3f}, {difference2:.3f}'.format(
            sequence=o['seq'], success=o['success'], precision=o['precision'],
            difference1=o['diff1'], difference2=o['diff2']))
"""

TC128_by_attribute = dict()
for result in results_TC128:
    for attribute in result.attributes:
        if attribute not in TC128_by_attribute.keys():
            TC128_by_attribute[attribute] = list()
        TC128_by_attribute[attribute].append(result)

TC128_by_attribute = OrderedDict(sorted(TC128_by_attribute.items(), key=lambda item: item[0]))

for attribute in TC128_by_attribute.keys():
    TC128_by_attribute[attribute] = sorted(TC128_by_attribute[attribute],
                                           key=lambda result: (result.tracker, result.sequence))

for attribute in TC128_by_attribute.keys():
    plot_success(results=TC128_by_attribute[attribute],
                 title='Success Plot on {dataset} for {attribute} using OPE'
                 .format(dataset='TC128', attribute=attribute))
    plot_precision(results=TC128_by_attribute[attribute],
                   title='Precision Plot on {dataset} for {attribute} using OPE'
                   .format(dataset='TC128', attribute=attribute))
    show_results(results=TC128_by_attribute[attribute],
                 title='Results on {dataset} for {attribute} using OPE'
                 .format(dataset='TC128', attribute=attribute))

plot_success(results=results_TC128,
             title='Success Plot on {dataset} using OPE'
             .format(dataset='TC128'))
plot_precision(results=results_TC128,
               title='Precision Plot on {dataset} using OPE'
               .format(dataset='TC128'))
show_results(results=results_TC128,
             title='Results on {dataset} using OPE'
             .format(dataset='TC128'))

# IZTECH15TC
results_IZTECH15TC = list()
for result in results_TC128:
    if (result.sequence in IZTECH15TC_SEQUENCES) and (result.dataset == 'TC128'):
        result.dataset = 'IZTECH15TC'
        results_IZTECH15TC.append(result)

results_IZTECH15TC = sorted(results_IZTECH15TC, key=lambda result: (result.tracker, result.sequence))

results_IZTECH15TC_by_attribute = dict()
for result in results_IZTECH15TC:
    for attribute in result.attributes:
        if attribute not in results_IZTECH15TC_by_attribute.keys():
            results_IZTECH15TC_by_attribute[attribute] = list()
        results_IZTECH15TC_by_attribute[attribute].append(result)

results_IZTECH15TC_by_attribute = OrderedDict(sorted(results_IZTECH15TC_by_attribute.items(), key=lambda item: item[0]))

for attribute in results_IZTECH15TC_by_attribute.keys():
    results_IZTECH15TC_by_attribute[attribute] = sorted(results_IZTECH15TC_by_attribute[attribute],
                                                        key=lambda result: (result.tracker, result.sequence))

for attribute in results_IZTECH15TC_by_attribute.keys():
    plot_success(results=results_IZTECH15TC_by_attribute[attribute],
                 title='Success Plot on {dataset} for {attribute} using OPE'
                 .format(dataset='IZTECH15TC', attribute=attribute))
    plot_precision(results=results_IZTECH15TC_by_attribute[attribute],
                   title='Precision Plot on {dataset} for {attribute} using OPE'
                   .format(dataset='IZTECH15TC', attribute=attribute))
    show_results(results=results_IZTECH15TC_by_attribute[attribute],
                 title='Results on {dataset} for {attribute} using OPE'
                 .format(dataset='IZTECH15TC', attribute=attribute))

plot_success(results=results_IZTECH15TC,
             title='Success Plot on {dataset} using OPE'
             .format(dataset='IZTECH15TC'))
plot_precision(results=results_IZTECH15TC,
               title='Precision Plot on {dataset} using OPE'
               .format(dataset='IZTECH15TC'))
show_results(results=results_IZTECH15TC,
             title='Results on {dataset} using OPE'
             .format(dataset='IZTECH15TC'))

# IZTECH15OTB
results_IZTECH15OTB = list()
for result in results_OTB100:
    if (result.sequence in IZTECH15OTB_SEQUENCES) and (result.dataset == 'OTB100'):
        result.dataset = 'IZTECH15OTB'
        results_IZTECH15OTB.append(result)

results_IZTECH15OTB = sorted(results_IZTECH15OTB, key=lambda result: (result.tracker, result.sequence))

results_IZTECH15OTB_by_attribute = dict()
for result in results_IZTECH15OTB:
    for attribute in result.attributes:
        if attribute not in results_IZTECH15OTB_by_attribute.keys():
            results_IZTECH15OTB_by_attribute[attribute] = list()
        results_IZTECH15OTB_by_attribute[attribute].append(result)

results_IZTECH15OTB_by_attribute = OrderedDict(
    sorted(results_IZTECH15OTB_by_attribute.items(), key=lambda item: item[0]))

for attribute in results_IZTECH15OTB_by_attribute.keys():
    results_IZTECH15OTB_by_attribute[attribute] = sorted(results_IZTECH15OTB_by_attribute[attribute],
                                                         key=lambda result: (result.tracker, result.sequence))

for attribute in results_IZTECH15OTB_by_attribute.keys():
    plot_success(results=results_IZTECH15OTB_by_attribute[attribute],
                 title='Success Plot on {dataset} for {attribute} using OPE'
                 .format(dataset='IZTECH15OTB', attribute=attribute))
    plot_precision(results=results_IZTECH15OTB_by_attribute[attribute],
                   title='Precision Plot on {dataset} for {attribute} using OPE'
                   .format(dataset='IZTECH15OTB', attribute=attribute))
    show_results(results=results_IZTECH15OTB_by_attribute[attribute],
                 title='Results on {dataset} for {attribute} using OPE'
                 .format(dataset='IZTECH15OTB', attribute=attribute))

plot_success(results=results_IZTECH15OTB,
             title='Success Plot on {dataset} using OPE'
             .format(dataset='IZTECH15OTB'))
plot_precision(results=results_IZTECH15OTB,
               title='Precision Plot on {dataset} using OPE'
               .format(dataset='IZTECH15OTB'))
show_results(results=results_IZTECH15OTB,
             title='Results on {dataset} using OPE'
             .format(dataset='IZTECH15OTB'))

# Overall
show_results_overall(results=results_combined,
                     title='Overall Results on Datasets using OPE')
