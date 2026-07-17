# -*- coding: utf-8 -*-
"""F0-6 能量历史和人工能量比 QA 测试（不依赖 Abaqus）。"""

from __future__ import print_function

import json
import os
import sys

import numpy as np

sys.modules['abaqusConstants'] = None
sys.modules['odbAccess'] = None
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
POSTPROCESS_DIR = os.path.join(REPO, 'Postprocess')
if POSTPROCESS_DIR not in sys.path:
    sys.path.insert(0, POSTPROCESS_DIR)
import Postprocess_All_surface_v2 as post


def main():
    time = np.linspace(0.0, 1.0, 11)
    base = {
        'ALLAE': np.full(time.size, 0.01), 'ALLIE': np.full(time.size, 1.0),
        'ALLKE': np.full(time.size, 0.2), 'ALLWK': np.full(time.size, 2.0),
        'ETOTAL': np.full(time.size, 0.01), 'ALLVD': np.full(time.size, 0.1),
    }
    result = post.energy_qa_payload({'time': time, 'values': base}, {})
    assert result['status'] == 'passed' and result['artificial_energy_ratio'] == 0.01
    bad = dict(base)
    bad['ALLAE'] = np.full(time.size, 0.2)
    assert post.energy_qa_payload({'time': time, 'values': bad}, {})['status'] == 'failed'
    assert post.energy_qa_payload({'time': time, 'values': {'ALLIE': base['ALLIE']}}, {})['status'] == 'not_available'

    payload = {}
    manifest = []
    post._put_raw_timeseries_payload(payload, manifest, {
        'ricker': {'energy_time': time, 'energy_values': base,
                   'qa_theory_json': json.dumps({'status': 'baseline_only'})}
    })
    assert 'raw_ricker_energy_ALLAE' in payload
    assert payload['raw_ricker_energy_ALLWK'].shape == (11,)
    print('test_energy_history_v2: 4/4 ok')


if __name__ == '__main__':
    main()
