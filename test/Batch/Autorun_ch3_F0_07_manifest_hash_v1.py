# -*- coding: utf-8 -*-
"""F0-7：校验论文 Autorun 清单中的脚本/输入哈希与实际文件一致。"""

from __future__ import print_function

import datetime
import glob
import hashlib
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'test', 'Abaqus')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _candidate_paths(manifest_path, key, manifest):
    candidates = []
    if os.path.isabs(str(key)):
        candidates.append(str(key))
    candidates.append(os.path.join(REPO_ROOT, str(key).replace('/', os.sep)))
    candidates.append(os.path.join(os.path.dirname(manifest_path), os.path.basename(str(key))))
    candidates.extend(glob.glob(os.path.join(os.path.dirname(manifest_path), '**', os.path.basename(str(key))), recursive=True))
    for field in ('reference_script',):
        if manifest.get(field):
            candidates.append(str(manifest[field]))
    fixed = [
        os.path.join(REPO_ROOT, 'Modeling', 'slope_frame_ssi_full_v2.py'),
        os.path.join(REPO_ROOT, 'Postprocess', 'Postprocess_All_surface_v2.py'),
        os.path.join(REPO_ROOT, 'Postprocess', 'Collect_All_results_v2.py'),
        os.path.join(REPO_ROOT, 'Postprocess', 'Plot_Hybrid_surface_v2.py'),
        os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt'),
        os.path.join(REPO_ROOT, 'Modeling', 'Archived', 'Hybrid', 'reference_layered_psv_v1.py'),
    ]
    candidates.extend([path for path in fixed if os.path.basename(path) == os.path.basename(str(key))])
    return list(dict.fromkeys(candidates))


def verify_manifest(path):
    manifest = load_json(path)
    hashes = dict(manifest.get('source_sha256') or {})
    if manifest.get('reference_sha256') and manifest.get('reference_script'):
        hashes[manifest['reference_script']] = manifest['reference_sha256']
    if not hashes:
        raise ValueError('清单缺少 source_sha256/reference_sha256：%s' % path)
    checks = []
    for key, expected in hashes.items():
        actual_path = next((candidate for candidate in _candidate_paths(path, key, manifest)
                            if os.path.isfile(candidate)), None)
        if actual_path is None:
            raise ValueError('清单哈希对应文件不存在：%s (%s)' % (key, path))
        actual = sha256(actual_path)
        checks.append({'key': key, 'path': actual_path, 'expected': expected, 'actual': actual,
                       'passed': actual == expected})
        if actual != expected:
            raise ValueError('哈希不一致：%s expected=%s actual=%s' % (key, expected, actual))
    run_dir = manifest.get('run_dir')
    if run_dir and not os.path.isdir(run_dir):
        raise ValueError('清单 run_dir 不存在：%s' % run_dir)
    return {'manifest': os.path.abspath(path), 'run_dir': run_dir, 'checks': checks, 'status': 'passed'}


def main():
    paths = [os.path.abspath(path) for path in sys.argv[1:] if path.lower().endswith('.json')]
    if not paths:
        paths = sorted(glob.glob(os.path.join(DEFAULT_ROOT, 'ch3_F0_*', 'run-*', '*manifest*.json')))
    if not paths:
        raise RuntimeError('未找到待审计的 F0 manifest')
    reports = [verify_manifest(path) for path in paths]
    output = {'status': 'passed', 'unit': 'F0-7', 'created_at': datetime.datetime.now().isoformat(),
              'manifest_count': len(reports), 'reports': reports}
    out_dir = os.path.dirname(reports[0]['manifest'])
    out_path = os.path.join(out_dir, 'f0_7_manifest_hash_report.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write('\n')
    print('F0-7 通过：%s' % json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
