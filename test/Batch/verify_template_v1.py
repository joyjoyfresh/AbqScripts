# -*- coding: utf-8 -*-
"""生产批处理入口通用模板 (Autorun_template_v2.py) 的自动化验证脚本。

通过单元测试和集成逻辑测试验证以下核心逻辑：
1. 配置白名单校验：未知顶层配置键必须拦截，白名单合法键必须通过。
2. 临界角校验预检：大于临界角时硬性拦截拒绝建模，小于时通过。
3. 散列计算一致性：配置字典散列与文件散列正确无误。
4. 解释器分发模拟：Modeling与后处理脚本准确映射到 Abaqus Python 与普通 Python。
5. 清理逻辑有效性：仅在 QA 通过且最终数据包存在时才执行清理，失败时完整保留。
6. 作业进度日志可增量读取，且不会重复输出旧进度。
"""

from __future__ import print_function

import os
import sys
import shutil
import tempfile
import unittest

# 将 Batch 目录加入 sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(REPO_ROOT, 'Batch'))

try:
    import Autorun_template_v2 as runner
except ImportError as e:
    print("错误：无法导入 Autorun_template_v2。导入错误: {}".format(str(e)))
    sys.exit(1)


class TestBatchTemplate(unittest.TestCase):

    def test_config_whitelist(self):
        """测试顶层键白名单校验。"""
        # 合法配置
        valid_cfg = {
            'material_cfg': {'angle': 15.0},
            'geometry_cfg': {'slope_angle': 30.0},
            'damping_cfg': {'enable': True},
            'run_cfg': {'critical_angle_check': True}
        }
        try:
            runner.validate_config(valid_cfg)
        except Exception as e:
            self.fail("合法配置未能通过校验: {}".format(str(e)))

        # 非法配置
        invalid_cfg = {
            'material_cfg': {'angle': 15.0},
            'some_illegal_cfg_key': {'value': 1.0}
        }
        with self.assertRaises(ValueError) as ctx:
            runner.validate_config(invalid_cfg)
        self.assertIn("发现了不属于白名单的非法顶层配置键", str(ctx.exception))

    def test_critical_angle_check(self):
        """测试临界角校验预检。"""
        # Poisson ratio = 0.3 下的临界角约为 32.3115°
        # 1. 超过临界角的入射（在启用拦截时应拦截）
        over_crit_cfg = {
            'material_cfg': {
                'angle': 35.0,
                'bedrock': {'vs': 2000.0, 'poisson_ratio': 0.3}
            },
            'run_cfg': {'critical_angle_check': True}
        }
        with self.assertRaises(ValueError) as ctx:
            runner.validate_config(over_crit_cfg)
        self.assertIn("达到或超过基岩临界角", str(ctx.exception))

        # 2. 超过临界角但显式关闭拦截（应该通过）
        over_crit_bypass = {
            'material_cfg': {
                'angle': 35.0,
                'bedrock': {'vs': 2000.0, 'poisson_ratio': 0.3}
            },
            'run_cfg': {'critical_angle_check': False}
        }
        try:
            runner.validate_config(over_crit_bypass)
        except Exception as e:
            self.fail("关闭临界角校验时应当允许通过: {}".format(str(e)))

        # 3. 未超过临界角的入射（应该通过）
        under_crit_cfg = {
            'material_cfg': {
                'angle': 15.0,
                'bedrock': {'vs': 2000.0, 'poisson_ratio': 0.3}
            },
            'run_cfg': {'critical_angle_check': True}
        }
        try:
            runner.validate_config(under_crit_cfg)
        except Exception as e:
            self.fail("未超临界角时应当通过: {}".format(str(e)))

    def test_hashing_consistency(self):
        """测试字典和文件哈希计算的一致性。"""
        d1 = {'a': 1, 'b': [2, 3]}
        d2 = {'b': [2, 3], 'a': 1}  # 改变键序
        h1 = runner.compute_dict_sha256(d1)
        h2 = runner.compute_dict_sha256(d2)
        self.assertEqual(h1, h2, "字典哈希计算应当对键序不敏感（内置排序）")

        # 临时文件哈希测试
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(temp_dir, 'test.txt')
            with open(test_file, 'wb') as f:
                f.write(b"Hello world!")
            h_file = runner.compute_file_sha256(test_file)
            self.assertEqual(len(h_file), 64, "应当生成64位十六进制 SHA-256 哈希值")
        finally:
            shutil.rmtree(temp_dir)

    def test_interpreter_dispatch(self):
        """测试 Abaqus 脚本与普通 Python 脚本的分发规则。"""
        # 验证 modeling 脚本在 ABAQUS_SCRIPTS 中
        self.assertIn('slope_frame_ssi_full_v2.py', runner.ABAQUS_SCRIPTS)
        # 验证后处理提取脚本在 ABAQUS_SCRIPTS 中
        self.assertIn('Postprocess_All_surface_v2.py', runner.ABAQUS_SCRIPTS)
        # 验证汇总与绘图脚本不在里面
        self.assertNotIn('Collect_All_results_v2.py', runner.ABAQUS_SCRIPTS)
        self.assertNotIn('Plot_Hybrid_surface_v2.py', runner.ABAQUS_SCRIPTS)

    def test_incremental_progress_log_reading(self):
        """测试建模进度日志的增量读取与旧日志拦截。"""
        temp_dir = tempfile.mkdtemp()
        try:
            log_path = os.path.join(temp_dir, 'slope_frame_ssi_full_v2.log')
            with open(log_path, 'wb') as handle:
                handle.write('2026-01-01 作业进度: job-old，已算到 1.000 秒/共 2.000 秒\n'.encode('utf-8'))
            old_time = 1000.0
            os.utime(log_path, (old_time, old_time))
            offset, messages = runner._read_new_job_progress(log_path, 0, old_time + 10.0)
            self.assertEqual(messages, [], "旧运行遗留日志不应转发到终端")

            with open(log_path, 'wb') as handle:
                handle.write('普通日志\n作业进度: job-a，已算到 2.000 秒/共 10.000 秒\n'.encode('utf-8'))
            current_time = old_time + 20.0
            os.utime(log_path, (current_time, current_time))
            offset, messages = runner._read_new_job_progress(log_path, 0, current_time)
            self.assertEqual(len(messages), 1)
            self.assertIn('job-a', messages[0])

            with open(log_path, 'ab') as handle:
                handle.write('作业进度: job-a，已算到 4.000 秒/共 10.000 秒\n'.encode('utf-8'))
            new_offset, messages = runner._read_new_job_progress(log_path, offset, current_time)
            self.assertGreater(new_offset, offset)
            self.assertEqual(len(messages), 1)
            self.assertIn('4.000', messages[0])
        finally:
            shutil.rmtree(temp_dir)

    def test_conditional_cleanup(self):
        """测试条件大文件清理规则。"""
        temp_dir = tempfile.mkdtemp()
        try:
            # 模拟工况目录
            odb_file = os.path.join(temp_dir, 'job-test.odb')
            npz_file = os.path.join(temp_dir, 'surface_results.npz')

            # 1. 成功运行且存在 NPZ：应被清理
            with open(odb_file, 'wb') as f: f.write(b"odb content")
            with open(npz_file, 'wb') as f: f.write(b"npz content")
            runner.delete_files_by_type(temp_dir, ['.odb'], run_ok=True)
            self.assertFalse(os.path.exists(odb_file), "运行成功且产物存在时，中间 odb 文件应当被删除")
            self.assertTrue(os.path.exists(npz_file), "最终 NPZ 产物应当被保留")

            # 2. 执行失败 (run_ok=False)：即使有 ODB，应被保留
            with open(odb_file, 'wb') as f: f.write(b"odb content")
            runner.delete_files_by_type(temp_dir, ['.odb'], run_ok=False)
            self.assertTrue(os.path.exists(odb_file), "运行失败时应当保留 odb 文件供诊断")

            # 3. 产物缺失：即使 run_ok=True，也应该被保留以供诊断
            os.remove(npz_file)  # 删去 NPZ
            runner.delete_files_by_type(temp_dir, ['.odb'], run_ok=True)
            self.assertTrue(os.path.exists(odb_file), "缺失最终产物时应当保留 odb 文件")

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    print("====== 开始运行通用批处理模板校验测试 ======")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchTemplate)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("====== 通用批处理模板校验通过 ======")
    sys.exit(0)
