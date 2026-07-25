# -*- coding: utf-8 -*-
"""H1—H3正式批处理入口的纯Python验证。"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER_PATH = os.path.join(
    REPO_ROOT, 'Run', 'Auto_ch4', 'Autorun_ch4_H_v1.py',
)
SPEC = importlib.util.spec_from_file_location('autorun_ch4_h_v1', RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TestChapter4HomogeneousRunner(unittest.TestCase):

    def test_case_counts_and_ids(self):
        """验证H1/H2/H3数量、ID唯一性和预生产子集。"""
        cases = RUNNER.build_case_plan()
        counts = {
            pool: len([item for item in cases if item['pool_id'] == pool])
            for pool in ('H1', 'H2', 'H3')
        }
        self.assertEqual(counts, {'H1': 35, 'H2': 10, 'H3': 6})
        self.assertEqual(len(set(item['case_id'] for item in cases)), 51)
        self.assertEqual(
            RUNNER.PILOT_CASE_IDS,
            ('H1-000035', 'H2-000006', 'H3-000006'),
        )
        self.assertEqual(
            len([item for item in cases if item['case_id'] in RUNNER.PILOT_CASE_IDS]),
            3,
        )
        self.assertEqual(RUNNER.PILOT_MAX_WORKERS, 3)
        self.assertEqual(RUNNER.PRODUCTION_MAX_WORKERS, 4)

    def test_pilot_and_remaining_worker_config(self):
        """验证首批3槽并行、剩余批次4槽并行。"""
        original_workers = RUNNER.pipeline.MAX_WORKERS
        try:
            RUNNER.configure_pipeline(RUNNER.PILOT_MAX_WORKERS)
            self.assertEqual(RUNNER.pipeline.MAX_WORKERS, 3)
            RUNNER.configure_pipeline(RUNNER.PRODUCTION_MAX_WORKERS)
            self.assertEqual(RUNNER.pipeline.MAX_WORKERS, 4)
        finally:
            RUNNER.pipeline.MAX_WORKERS = original_workers

    def test_frozen_production_config(self):
        """验证标准激励、频带、尾段、初态和物理边界。"""
        cases = RUNNER.build_case_plan()
        for case in cases:
            config = case['config']
            self.assertEqual(config['material_cfg']['layers'], [])
            self.assertLessEqual(config['material_cfg']['angle'], 30.0)
            self.assertEqual(config['damping_cfg']['fc'], 4.0)
            self.assertEqual(config['time_cfg']['tail_seconds'], 6.0)
            self.assertEqual(
                config['freefield_cfg']['initial_state_mode'], 'incremental',
            )
            self.assertEqual(
                config['run_cfg']['qa_cfg']['frf_fmax_hz'], 12.0,
            )
            self.assertEqual(
                config['geometry_cfg']['side_clearance'],
                RUNNER.PRODUCTION_SIDE_CLEARANCE,
            )
            self.assertIn('energy', config['run_cfg']['qa_cfg']['required'])
            self.assertEqual(
                config['run_cfg']['wave_files'], [RUNNER.WAVE_FILENAME],
            )
            self.assertFalse(config['eql_cfg']['enable'])
            self.assertEqual(config['tssi_cfg']['scene'], 'freefield')
            if case['case_id'] in RUNNER.PILOT_CASE_IDS:
                qa_cfg = config['run_cfg']['qa_cfg']
                self.assertIn('domain', qa_cfg['required'])
                self.assertEqual(qa_cfg['mode'], 'window_convergence')
                self.assertEqual(qa_cfg['tol'], 0.02)
                self.assertEqual(qa_cfg['min_points'], 790)
                self.assertIn('run-001', qa_cfg['reference_npz'])

    def test_h2_prototypes_are_in_h1(self):
        """验证H2尺度原型均来自H1几何库。"""
        cases = RUNNER.build_case_plan()
        h1_pairs = set(
            (item['slope_angle'], item['incident_angle'])
            for item in cases if item['pool_id'] == 'H1'
        )
        self.assertTrue(set(RUNNER.H2_PROTOTYPES).issubset(h1_pairs))

    def test_prepare_only_artifacts(self):
        """验证预生产只生成目录、配置和清单，不生成求解产物。"""
        root_dir = tempfile.mkdtemp()
        registry_dir = tempfile.mkdtemp()
        original_registry_dir = RUNNER.REGISTRY_DIR
        original_master_path = RUNNER.MASTER_MANIFEST_PATH
        try:
            RUNNER.configure_pipeline()
            RUNNER.REGISTRY_DIR = registry_dir
            RUNNER.MASTER_MANIFEST_PATH = os.path.join(
                registry_dir, 'master_manifest.csv',
            )
            cases = RUNNER.build_case_plan()
            sources = RUNNER.source_files()
            statuses = dict(
                (case['folder_name'], 'planned') for case in cases
            )
            RUNNER.prepare_run(root_dir, cases, sources, statuses)
            self.assertTrue(os.path.isfile(
                os.path.join(root_dir, 'run_manifest.json'),
            ))
            self.assertTrue(os.path.isfile(
                os.path.join(root_dir, 'H1_H3预生产清单.json'),
            ))
            self.assertEqual(
                len([
                    name for name in os.listdir(root_dir)
                    if name.startswith('case-H')
                ]),
                51,
            )
            self.assertEqual(
                len(list(os.scandir(root_dir))), 53,
            )
            self.assertFalse(any(
                name.endswith('.odb')
                for name in os.listdir(os.path.join(root_dir, 'case-H1-000001'))
            ))
            with open(
                os.path.join(root_dir, 'H1_H3预生产清单.json'),
                'r', encoding='utf-8',
            ) as handle:
                preparation = json.load(handle)
            self.assertFalse(preparation['abaqus_started'])
            self.assertEqual(preparation['case_count'], 51)
            self.assertEqual(preparation['pilot_model_workers'], 3)
            self.assertEqual(preparation['remaining_model_workers'], 4)
            self.assertEqual(len(preparation['source_file_sha256']), 6)
            self.assertEqual(len(preparation['case_config_sha256']), 51)
            RUNNER.validate_prepared_run(root_dir, cases, sources)
        finally:
            RUNNER.REGISTRY_DIR = original_registry_dir
            RUNNER.MASTER_MANIFEST_PATH = original_master_path
            shutil.rmtree(root_dir)
            shutil.rmtree(registry_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
