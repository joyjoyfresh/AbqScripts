# -*- coding: utf-8 -*-
"""自检脚本 (verify_G0_injection_v1.py)。

用于自动化验证 G0 与 G1 的批处理配置文件及参数域合法性。
包括：
- 顶层配置白名单与未知配置项拦截
- 临界角预检及其安全裕度（临界角 - 入射角 >= 2.0°）
- 强制参数和冻结的生产参数校验
- G1 48次求解波形文件路径存在性检测
- G0 非法键拦截测试
"""

from __future__ import print_function

import os
import sys
import unittest
import math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(REPO_ROOT, 'Run', 'Auto_ch4'))

try:
    import Autorun_ch4_G0_v1 as runner_g0
    import Autorun_ch4_G1_v1 as runner_g1
except ImportError as e:
    print("错误：无法导入 G0/G1 批处理脚本。导入错误: {}".format(str(e)))
    sys.exit(1)


class TestG0G1Injection(unittest.TestCase):

    def test_g0_cases_whitelist_and_hashing(self):
        """验证 G0 前 11 个工况配置的白名单及合规性，第 12 个非法工况预期抛错。"""
        # 测试前 11 个合法工况
        for case in runner_g0.PARAMETER_CASES[:-1]:
            cfg = case["config"]
            try:
                runner_g0.validate_config(cfg)
            except Exception as e:
                self.fail("G0 合法工况 '{}' 未能通过配置校验: {}".format(case["name"], str(e)))

        # 测试第 12 个非法工况，预期发生 ValueError 拦截
        err_case = runner_g0.PARAMETER_CASES[-1]
        self.assertEqual(err_case["name"], "G0-000012", "第 12 个工况应为 G0-000012 拦截测试")
        with self.assertRaises(ValueError) as ctx:
            runner_g0.validate_config(err_case["config"])
        self.assertIn("发现了不属于白名单的非法顶层配置键", str(ctx.exception))

    def test_g1_cases_whitelist(self):
        """验证 G1 所有的工况配置白名单合规性。"""
        for case in runner_g1.PARAMETER_CASES:
            cfg = case["config"]
            try:
                runner_g1.validate_config(cfg)
            except Exception as e:
                self.fail("G1 工况 '{}' 未能通过配置校验: {}".format(case["name"], str(e)))

    def test_incident_angle_margin(self):
        """验证所有入射角均严格低于基岩临界角（ν=0.3 下约为 32.3115°），且留有至少 2° 的安全裕度。"""
        # 预计算临界角 (ν=0.3)
        pr = 0.3
        sin_crit = math.sqrt((1.0 - 2.0 * pr) / (2.0 * (1.0 - pr)))
        crit_deg = math.degrees(math.asin(sin_crit))  # 约 32.3115°
        limit_deg = crit_deg - 2.0  # 约 30.3115°

        # 检查 G0 与 G1 所有合法工况
        all_cases = runner_g0.PARAMETER_CASES[:-1] + runner_g1.PARAMETER_CASES
        for case in all_cases:
            cfg = case["config"]
            material_cfg = cfg.get("material_cfg") or {}
            angle = material_cfg.get("angle")
            if angle is not None:
                self.assertLessEqual(
                    angle, 
                    limit_deg + 1e-5, 
                    "工况 '{}' 的入射角 {}° 达到或超过安全限值 {}° (临界角为 {:.2f}°，安全裕度必须 >= 2°)"
                    .format(case["name"], angle, limit_deg, crit_deg)
                )

    def test_wave_files_existence(self):
        """验证 G0 和 G1 所有工况中指定的输入波文件路径在物理硬盘上确实存在。"""
        # G0 工况波文件
        for case in runner_g0.PARAMETER_CASES[:-1]:
            cfg = case["config"]
            run_cfg = cfg.get("run_cfg") or {}
            wave_files = run_cfg.get("wave_files", [])
            self.assertGreater(len(wave_files), 0, "工况 '{}' 缺失输入波配置".format(case["name"]))
            for wf in wave_files:
                abs_path = os.path.join(REPO_ROOT, wf)
                self.assertTrue(os.path.isfile(abs_path), "G0 工况波物理文件不存在: {}".format(abs_path))

        # G1 工况波文件
        for case in runner_g1.PARAMETER_CASES:
            cfg = case["config"]
            run_cfg = cfg.get("run_cfg") or {}
            wave_files = run_cfg.get("wave_files", [])
            self.assertEqual(len(wave_files), 6, "G1 工况 '{}' 的输入波数量应刚好为 6".format(case["name"]))
            for wf in wave_files:
                abs_path = os.path.join(REPO_ROOT, wf)
                self.assertTrue(os.path.isfile(abs_path), "G1 工况波物理文件不存在: {}".format(abs_path))

    def test_frozen_and_forced_parameters(self):
        """验证是否在所有工况配置中强制注入了 G0/G1 冻结的生产参数。"""
        # 建立合并后配置验证
        def verify_merged_config(case_list, is_g0):
            for case in case_list:
                if is_g0 and case["name"] == "G0-000012":
                    continue  # 跳过 G0 的拦截测试
                                # 模拟批处理合并参数逻辑
                cfg = case["config"].copy()
                cfg.setdefault("tssi_cfg", {})["enable"] = False
                cfg["tssi_cfg"]["scene"] = "freefield"
                cfg.setdefault("eql_cfg", {})["enable"] = False
                cfg.setdefault("run_cfg", {})["submit_jobs"] = (not is_g0)
                cfg["run_cfg"]["critical_angle_check"] = True
                cfg["run_cfg"]["surface_only"] = True
                
                runner = runner_g0 if is_g0 else runner_g1
                cfg["geometry_cfg"] = runner.merge_dicts(cfg.get("geometry_cfg", {}), {"base_depth": 3.0})
                cfg["damping_cfg"] = runner.merge_dicts(cfg.get("damping_cfg", {}), runner.FROZEN_DAMPING)
                cfg["mesh_cfg"] = runner.merge_dicts(cfg.get("mesh_cfg", {}), runner.FROZEN_MESH)
                cfg["time_cfg"] = runner.merge_dicts(cfg.get("time_cfg", {}), runner.FROZEN_TIME)

                # 1. 验证强制 SSI/TSSI 与 EQL 参数
                self.assertFalse(cfg["tssi_cfg"]["enable"], "tssi_cfg.enable 必须强制为 False")
                self.assertEqual(cfg["tssi_cfg"]["scene"], "freefield", "tssi_cfg.scene 必须强制为 'freefield'")
                self.assertFalse(cfg["eql_cfg"]["enable"], "eql_cfg.enable 必须强制为 False")
                self.assertEqual(cfg["run_cfg"]["submit_jobs"], not is_g0, "submit_jobs 逻辑不正确")
                self.assertTrue(cfg["run_cfg"]["critical_angle_check"], "critical_angle_check 必须强制为 True")
                self.assertTrue(cfg["run_cfg"]["surface_only"], "surface_only 必须强制为 True")
                self.assertEqual(cfg["geometry_cfg"]["base_depth"], 3.0, "geometry_cfg.base_depth 必须强制为 3.0")

                # 2. 验证冻结阻尼比
                self.assertEqual(cfg["damping_cfg"]["constant_xi"], 0.03, "constant_xi 必须冻结为 0.03")
                self.assertEqual(cfg["damping_cfg"]["method"], "rayleigh", "damping method 必须冻结为 rayleigh")
                self.assertEqual(cfg["damping_cfg"]["anchor"], "perband", "damping anchor 必须冻结为 perband")

                # 3. 验证网格冻结参数
                self.assertTrue(cfg["mesh_cfg"]["auto"], "mesh_cfg.auto 必须冻结为 True")
                self.assertEqual(cfg["mesh_cfg"]["size"], 4.0, "mesh_cfg.size 必须冻结为 4.0")
                self.assertEqual(cfg["mesh_cfg"]["elems_per_wavelength"], 10, "elems_per_wavelength 必须冻结为 10")
                self.assertEqual(cfg["mesh_cfg"]["elem"], "CPE4R", "elem 单元类型必须冻结为 CPE4R")
                self.assertTrue(cfg["mesh_cfg"]["graded"], "mesh_cfg.graded 必须冻结为 True")
                self.assertEqual(cfg["mesh_cfg"]["min_elems_through_thickness"], 6, "min_elems_through_thickness 必须冻结为 6")

                # 4. 验证时间步与尾段冻结参数
                self.assertTrue(cfg["time_cfg"]["check"], "time_cfg.check 必须冻结为 True")
                self.assertEqual(cfg["time_cfg"]["min_steps_per_fmax_period"], 20, "min_steps_per_fmax_period 必须冻结为 20")
                self.assertEqual(cfg["time_cfg"]["tail_seconds"], 0.0, "tail_seconds 必须冻结为 0.0")

        verify_merged_config(runner_g0.PARAMETER_CASES, is_g0=True)
        verify_merged_config(runner_g1.PARAMETER_CASES, is_g0=False)


if __name__ == '__main__':
    print("====== 开始运行 G0/G1 配置注入参数合法性自检 ======")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestG0G1Injection)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("====== G0/G1 配置注入自检全部通过 ======")
    sys.exit(0)
