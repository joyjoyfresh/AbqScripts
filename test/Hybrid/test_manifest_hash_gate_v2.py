# -*- coding: utf-8 -*-
"""F0-7 清单哈希门禁及故意篡改负对照测试。"""

from __future__ import print_function

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BATCH_DIR = os.path.join(REPO, 'Batch')
if BATCH_DIR not in sys.path:
    sys.path.insert(0, BATCH_DIR)
import Autorun_ch3_F0_07_manifest_hash_v1 as gate


def main():
    source = os.path.join(REPO, 'Modeling', 'Hybrid', 'reference_layered_psv_v1.py')
    with tempfile.TemporaryDirectory() as folder:
        run_dir = os.path.join(folder, 'run-001')
        os.makedirs(run_dir)
        manifest_path = os.path.join(run_dir, 'reference_manifest.json')
        manifest = {'run_dir': run_dir, 'reference_script': source,
                    'reference_sha256': gate.sha256(source)}
        with open(manifest_path, 'w') as fh:
            json.dump(manifest, fh)
        assert gate.verify_manifest(manifest_path)['status'] == 'passed'
        manifest['reference_sha256'] = '0' * 64
        with open(manifest_path, 'w') as fh:
            json.dump(manifest, fh)
        failed = False
        try:
            gate.verify_manifest(manifest_path)
        except ValueError:
            failed = True
        assert failed
    print('test_manifest_hash_gate_v2: 2/2 ok')


if __name__ == '__main__':
    main()
