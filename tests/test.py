from __future__ import absolute_import

import os

from got10k.experiments import ExperimentTColor128, ExperimentOTB

from tests.adaptive_kalman_siam_distinct import AdaptiveKalmanSiamDistinct
from tests.adaptive_kalman_siam_p import AdaptiveKalmanSiamP
from tests.adaptive_kalman_siam_single import AdaptiveKalmanSiamSingle
from tests.kalman_siam import KalmanSiam
from tests.siamfc import SiamFC

paths = dict()
paths['root'] = os.path.expanduser('/content')
paths['datasets'] = os.path.join(paths['root'], 'datasets')
paths['iztech15tc'] = os.path.join(paths['datasets'], 'iztech15tc')
paths['iztech15otb'] = os.path.join(paths['datasets'], 'iztech15otb')
paths['results'] = os.path.join(paths['root'], 'results')
paths['reports'] = os.path.join(paths['root'], 'reports')
paths['weights'] = os.path.join(paths['root'], 'weights')
paths['weight'] = os.path.join(paths['weights'], 'siamfc_alexnet_e50.pth')


def test_trackers():
    trackers = [
        SiamFC(net_path=paths['weight']),
        KalmanSiam(net_path=paths['weight']),
        AdaptiveKalmanSiamP(net_path=paths['weight']),
        AdaptiveKalmanSiamSingle(net_path=paths['weight']),
        AdaptiveKalmanSiamDistinct(net_path=paths['weight']),
    ]

    experiments = [
        ExperimentOTB(
            root_dir=paths['iztech15otb'],
            result_dir=paths['results'],
            report_dir=paths['reports']
        ),
        ExperimentTColor128(
            root_dir=paths['iztech15tc'],
            result_dir=paths['results'],
            report_dir=paths['reports']
        )
    ]

    for tracker in trackers:
        for experiment in experiments:
            experiment.run(tracker)
            experiment.report([tracker.name])


test_trackers()
