import json

from performance.common import OTB100_SEQUENCES_ATTRIBUTES, TC128_SEQUENCES_ATTRIBUTES


class Result:
    def __init__(self, tracker: str = '', dataset: str = '', sequence: str = '', groundtruth: str = '', successes=None,
                 precisions=None, attributes=None, fps: float = 0, success: float = 0, precision: float = 0):
        if successes is None:
            successes = list()
        if precisions is None:
            precisions = list()
        if attributes is None:
            attributes = list()
        self.tracker = tracker
        self.dataset = dataset
        self.sequence = sequence
        self.groundtruth = groundtruth
        self.successes = successes
        self.precisions = precisions
        self.fps = fps
        self.success = success
        self.precision = precision
        self.attributes = attributes

    def __repr__(self):
        return '-'.join([self.tracker, self.dataset, self.sequence, self.groundtruth])

    def __str__(self):
        return '-'.join([self.tracker, self.dataset, self.sequence, self.groundtruth])


def load_results(file_path, trackers=None, datasets=None, sequences=None, attributes=None):
    if trackers is None:
        trackers = list()
    if datasets is None:
        datasets = list()
    if sequences is None:
        sequences = list()
    if attributes is None:
        attributes = list()
    results = list()
    try:
        data = dict()
        with open(file=file_path, encoding='UTF-8', mode='r') as result_file:
            data = result_file.read()
            data = json.loads(data)
        for dataset in data.keys():
            if datasets is None or len(datasets) == 0 or dataset in datasets:
                for tracker in data[dataset].keys():
                    if trackers is None or len(trackers) == 0 or tracker in trackers:
                        for sequence in data[dataset][tracker]['sequences'].keys():
                            groundtruth = '1'
                            sequence_parsed = sequence
                            if len(sequence.split('.')) > 1:
                                groundtruth = sequence.split('.')[-1]
                                sequence_parsed = ''.join(sequence.split('.')[:-1])
                            if sequences is None or len(sequences) == 0 or sequence in sequences:
                                sequence_attributes = list()
                                if dataset == 'OTB100':
                                    for sequence_with_attributes in OTB100_SEQUENCES_ATTRIBUTES:
                                        sequence_name, sequence_attributes = sequence_with_attributes
                                        if sequence_name == sequence_parsed \
                                                and (attributes is None or len(attributes) == 0 or any(
                                            attribute in sequence_attributes for attribute in attributes)):
                                            sequence_attributes = sorted(sequence_attributes)
                                            result = Result(
                                                tracker=tracker,
                                                dataset=dataset,
                                                sequence=sequence_parsed,
                                                groundtruth=groundtruth,
                                                successes=data[dataset][tracker]['sequences'][sequence]['successes'],
                                                precisions=data[dataset][tracker]['sequences'][sequence]['precisions'],
                                                fps=data[dataset][tracker]['sequences'][sequence]['fps'],
                                                success=data[dataset][tracker]['sequences'][sequence]['success'],
                                                precision=data[dataset][tracker]['sequences'][sequence]['precision'],
                                                attributes=sequence_attributes
                                            )
                                            results.append(result)
                                if dataset == 'TC128':
                                    for sequence_with_attributes in TC128_SEQUENCES_ATTRIBUTES:
                                        sequence_name, sequence_attributes = sequence_with_attributes
                                        if sequence_name == sequence_parsed \
                                                and (attributes is None or len(attributes) == 0 or any(
                                            attribute in sequence_attributes for attribute in attributes)):
                                            sequence_attributes = sorted(sequence_attributes)
                                            result = Result(
                                                tracker=tracker,
                                                dataset=dataset,
                                                sequence=sequence_parsed,
                                                groundtruth=groundtruth,
                                                successes=data[dataset][tracker]['sequences'][sequence]['successes'],
                                                precisions=data[dataset][tracker]['sequences'][sequence]['precisions'],
                                                fps=data[dataset][tracker]['sequences'][sequence]['fps'],
                                                success=data[dataset][tracker]['sequences'][sequence]['success'],
                                                precision=data[dataset][tracker]['sequences'][sequence]['precision'],
                                                attributes=sequence_attributes
                                            )
                                            results.append(result)
    except:
        print('Error')
    return results
