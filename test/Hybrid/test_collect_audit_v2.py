# -*- coding: utf-8 -*-
"""F0-5 收集器审计字段测试（不依赖 Abaqus）。"""

from __future__ import print_function

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
COLLECT_DIR = os.path.join(REPO, 'Postprocess', 'Hybrid')
if COLLECT_DIR not in sys.path:
    sys.path.insert(0, COLLECT_DIR)
import Collect_All_results_v2 as collect


def main():
    with tempfile.TemporaryDirectory() as folder:
        meta = {
            'model_type': 'single', 'model_script': 'slope_frame_ssi_full_v2.py',
            'mesh_size': 8.0, 'validation_geometry': 'flat',
            'geometry': {'total_L': 270.0, 'bedrock_thickness': 100.0, 'H': 0.0},
            'damping': {'enable': True, 'fc': 4.0, 'method': 'rayleigh', 'anchor': 'input'},
        }
        config = {'mesh_cfg': {'elem': 'CPE4R'}, 'run_cfg': {'wave_files': ['input.txt']}}
        with open(os.path.join(folder, 'case_meta.json'), 'w') as fh:
            json.dump(meta, fh)
        with open(os.path.join(folder, 'case_config.json'), 'w') as fh:
            json.dump(config, fh)
        with open(os.path.join(folder, 'input.txt'), 'wb') as fh:
            fh.write(b'0 0\n0.1 1\n')
        with open(os.path.join(folder, 'slope_frame_ssi_full_v2.log'), 'w') as fh:
            fh.write('网格统计: 单元=442, 节点=490\n')
        with open(os.path.join(folder, 'geometry_validation.json'), 'w') as fh:
            json.dump({'node_count': 490, 'element_count': 442,
                       'bbox': {'xmin': 0.0, 'xmax': 270.0, 'ymin': 0.0, 'ymax': 100.0}}, fh)
        summary = {
            'records': [{
                'record': 'ricker_wavelet_4Hz', 'AR_max': 1.2, 'suspect': False,
                'dt': 0.001, 'duration': 3.0, 'n_nodes': 35,
                'qa_required': ['theory', 'mesh'],
                'qa_gates': {'theory': True, 'reflection': False, 'mesh': True, 'time': False,
                             'domain': False, 'energy': False, 'external': False},
                'qa_gate_status': {'theory': 'passed', 'mesh': 'passed'},
                'overall_pass': True, 'qa_status': 'passed',
            }]
        }
        manifest = [{'name': 'sgrid_response_ricker_wavelet_4Hz-flat.csv', 'key': 'sgrid_response_ricker_wavelet_4Hz_flat'}]
        np.savez_compressed(os.path.join(folder, 'surface_results.npz'),
                            manifest_json=json.dumps(manifest), surface_summary_json=json.dumps(summary))
        records = collect._summary_from_surface_npz(os.path.join(folder, 'surface_results.npz'))
        record = records['ricker_wavelet_4Hz']
        audit = collect._audit_fields(folder, meta, config, record)
        assert audit['actual_dt'] == 0.001
        assert audit['n_model_nodes'] == 490 and audit['n_elements'] == 442
        assert audit['domain_xmax'] == 270.0 and audit['qa_theory'] is True
        assert audit['overall_pass'] is True
        hashes = json.loads(audit['script_hashes_json'])
        assert 'Modeling/Hybrid/slope_frame_ssi_full_v2.py' in hashes
    print('test_collect_audit_v2: 5/5 ok')


if __name__ == '__main__':
    main()
