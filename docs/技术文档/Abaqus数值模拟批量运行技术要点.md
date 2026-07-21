# Abaqus 批处理运行踩坑记录

## 1. 背景

在执行 P1 左右观测窗外扩诊断时，曾临时创建 `Batch/Autorun_P1_window_test_v1.py` 并尝试通过后台 `Start-Process` 与 Codex `shell_command` 启动 Abaqus 批处理。该过程暴露出两个流程问题：

1. 临时诊断性质的 Abaqus 批处理脚本不应放在 `Batch/`，应放在 `test/Batch/`；论文正文研究单元的 Autorun 属于正式研究脚本，统一放在 `Batch/`；
2. Abaqus 长时批处理不适合用短生命周期后台进程随手启动，否则容易只生成半截模型目录，未真正完成求解和后处理。

## 2. 脚本放置规则

以后凡是仅用于临时排查、且其结果不进入论文正文的 Abaqus 运行脚本，统一放在：

```text
test/Batch/
```

默认输出目录应放在：

```text
test/Abaqus/<测试名称>/
```

与论文正文相关的正式研究 Autorun 统一放在 `Batch/`，其 Abaqus 工况与结果统一放在 `Run/<研究单元>/run-###/`；脚本默认自动递增运行编号，避免覆盖既有证据。临时诊断脚本若需输出到 `Run/` 下，应通过命令行参数显式指定，例如：

```text
python C:\Users\12462\Documents\Code\AbqScripts\test\Batch\Autorun_P1_window_test_v1.py C:\Users\12462\Documents\Code\AbqScripts\Run\P1_window_test
```

这样可以避免把临时诊断脚本误认为正式生产批处理脚本，也便于后续清理测试产物。

## 3. 本次踩坑：后台启动 Abaqus 被截断

本次曾尝试使用类似方式后台启动：

```powershell
Start-Process -FilePath python -ArgumentList ... -RedirectStandardOutput ... -RedirectStandardError ...
```

结果表现为：

- 工况目录已创建；
- `case_config.json`、`case_meta.json`、`.cae`、`.jnl`、`.rec` 等文件生成；
- 建模日志只推进到“等效节点力计算完成”附近；
- 没有进入“提交作业”或“所有作业已完成”阶段；
- 没有生成 `job-*.odb` 或 `surface_results.npz`。

这说明后台进程在 Codex 工具调用结束或超时后被截断，导致 Abaqus 只执行了前半段建模流程。

## 4. 推荐运行方式

### 4.1 小规模快速检查

只检查配置生成、脚本导入和语法时，可以使用普通 Python：

```powershell
python -m py_compile test\Batch\Autorun_P1_window_test_v1.py
python -c "import sys; sys.path.insert(0, r'test\Batch'); import Autorun_P1_window_test_v1 as m; print(len(m.build_window_cases()))"
```

### 4.2 真正跑 Abaqus 求解

真正提交 Abaqus 作业时，优先采用前台命令，并给足超时时间：

```powershell
python test\Batch\Autorun_P1_window_test_v1.py C:\Users\12462\Documents\Code\AbqScripts\Run\P1_window_test
```

如果在 Codex 工具中运行，必须意识到 `timeout_ms` 到达后可能中止父进程或造成半截结果。对长时间 Abaqus 求解，更稳妥的做法是：

1. 将测试脚本准备好；
2. 用户在本机终端或 Abaqus 命令行中直接运行；
3. Codex 负责后续读取日志、检查产物和分析结果。

### 4.3 不推荐方式

不建议用一次性后台命令启动长时 Abaqus 求解，例如：

```powershell
Start-Process ... -WindowStyle Hidden
```

除非能确认该进程不会随工具调用生命周期被回收，并且有可靠的进程监控与日志刷新机制。

## 5. 判断一次 Abaqus 批处理是否真正完成

不要只看是否生成 `.cae` 或 `case_meta.json`。至少检查：

```text
slope_frame_ssi_full_v2.log 中出现 “提交作业” 或 “所有作业已完成”
job-*.odb 是否存在
Postprocess_All_surface_v2.log 是否存在
surface_response_*.csv 是否存在
surface_results.npz 是否存在
results/index.csv 是否被汇总生成
```

如果只生成 `.cae`、`.jnl`、`.rec`，但没有 `job-*.odb` 和响应 CSV，则只能说明建模阶段启动过，不能说明模拟完成。

### 5.1 长时作业的分析时间进度

`slope_frame_ssi_full_v2.py`现根据Abaqus/Standard的`job-*.sta`状态文件定时记录目标动力分析步的真实推进时间。默认配置为：

```json
{
  "run_cfg": {
    "job_progress_interval_seconds": 360.0
  }
}
```

日志格式示例：

```text
作业进度: job-example，已算到 12.345 秒/共 45.900 秒（26.9%，Step-earthquake）
```

该进度中的`x秒`来自`.sta`最新已收敛增量的`STEP TIME`，`y秒`来自模型中`Step-earthquake.timePeriod`，不是按墙钟耗时推算的预计完成时间。若存在重力步，监控仍只报告地震动力步；动力步尚未开始时显示`0/y`。`run_cfg.job_progress_interval_seconds<=0`可关闭建模日志的定时进度，但不改变求解过程。

`Batch/Autorun_template_v2.py`使用`subprocess.Popen`保持子进程输出落盘，同时默认每5分钟增量读取建模日志中的新`作业进度`行，并以以下格式转发到终端：

```text
[运行状态][case-example] 作业进度: job-example，已算到 12.345 秒/共 45.900 秒（26.9%，Step-earthquake）
```

多工况并发时每条进度使用终端输出锁整行打印，避免字符交叉。轮询只读取新增内容，并按启动时间忽略同目录上一次运行遗留的旧日志。默认情况下，建模日志每6分钟产生一条定时进度，终端每5分钟检查一次，因此新进度在终端中的显示可能比日志写入最多滞后约5分钟；子进程完成状态另以5秒间隔轻量检查，检测到结束后立即读取剩余进度，不会因低频进度显示而推迟后处理或下一个工况。`.sta`仍可在工况全部通过后按既有清理规则删除，但运行过程中不得提前删除。

当前G1r专用入口`Run/Auto_ch4/Autorun_ch4_G1r.py`已同步相同的终端转发逻辑，因此H1双波初态诊断也会直接显示上述进度；专用入口的工况筛选、并发上限和科学门禁未作改变。

## 6. 本次遗留状态

本次中断尝试可能留下以下半截目录：

```text
Run/P1_window_test/
Run/P1_window_test_full/
```

这些目录可能只包含建模中间文件，不能直接用于结果判断。若后续要重新运行，建议使用新的输出目录，或在确认无用后由用户手动清理。

**更新状态（2026-07-11）**：
已将原误放在 `Batch/` 下的 `Autorun_P1_window_test_v1.py` 移回到 `test/Batch/`，并修复了其在深层目录下的相对路径解析缺陷（将 `WORKSPACE_DIR` 退三级解析，修改默认输出目录为 `test/Abaqus/P1_window_test`）。导入与编译校验均已通过，可正常从 `test/Batch/` 启动。

## 7. 后续约定

1. 测试性质脚本一律放入 `test/Batch/`；
2. 测试产物默认写入 `test/Abaqus/`；
3. 正式批处理脚本才放入 `Batch/`；
4. 长时间 Abaqus 求解尽量由用户终端前台运行，Codex 负责准备脚本、检查日志和分析结果；
5. 判断完成必须以 ODB、后处理 CSV 和日志完成标志为准。
