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

`slope_frame_ssi_full_v2.py`现根据Abaqus/Standard的`job-*.sta`状态文件定时记录目标动力分析步的真实推进时间。监控线程会在`job.submit()`之前启动，避免Abaqus提交调用占住主线程后无法进入进度循环。默认配置为：

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

`Batch/Autorun_template_v2.py`使用`subprocess.Popen`保持子进程输出落盘，同时默认每5分钟直接读取本次运行最新`job-*.sta`，并结合建模日志中的模型名与动力步总时长生成以下终端状态：

```text
[运行状态][case-example] 作业进度: job-example，已算到 12.345 秒/共 45.900 秒（26.9%，Step-earthquake）
```

多工况并发时每条进度使用终端输出锁整行打印，避免字符交叉。Autorun只选择启动时间之后新建或更新的`.sta`，不会误报同目录上一次运行的状态；若`.sta`尚未产生或总时长暂未识别，才回退转发建模日志中的`作业进度`行。建模日志默认每6分钟更新，终端默认每5分钟直接采样求解器状态，两者互不依赖。子进程完成状态另以5秒间隔轻量检查，检测到结束后立即读取最终进度，不会因低频进度显示而推迟后处理或下一个工况。`.sta`仍可在工况全部通过后按既有清理规则删除，但运行过程中不得提前删除。

当前G1r专用入口`Run/Auto_ch4/Autorun_ch4_G1r.py`已同步相同的`.sta`直读逻辑，因此后续启动的H1双波初态诊断和正式真实波复算都会直接显示上述进度。源码修改不会热加载到已经启动的Autorun或已复制到运行目录的建模脚本，正在运行的批次保持原行为且无须中断。正式真实波复算必须在独立运行目录中进行；入口支持用`--release-evidence-root`和`--initial-diagnostic-root`分别导入组合宽频门与增量初态诊断门，并同时锁定门控文件、门控内容和源运行清单的SHA-256。后续执行`--run-real-only`时会再次核验这些证据，任何源证据变化都会阻止启动Abaqus。单机并发上限仍为4，单作业CPU和内存设置不变。

### 5.2 建模—单工况后处理流水线

通用模板和G1r正式入口将每个工况的本地流程拆成两个独立队列：`MODEL_SCRIPT_SEQUENCE`由建模线程池领取，`CASE_POSTPROCESS_SCRIPT_SEQUENCE`由单工况后处理线程池领取。某工况建模进程正常退出后，状态先写为`model_passed`并立即提交后处理；原建模槽同时领取下一个`planned`工况，不再等待前一工况的ODB提取完成。全局汇总和跨工况绘图仍在全部单工况后处理结束后执行，不能提前读取半成品。

当前正式配置为4个建模/求解槽和1个单工况后处理槽，因此最多同时存在4个求解流程与1个只读ODB后处理流程，但求解作业并发上限仍是4。建模使用`abaqus cae noGUI=...`，后处理按脚本约定使用更轻量的`abaqus python ...`。若建模失败，不提交对应后处理；若后处理失败，保留ODB和日志。清单按`model_running → model_passed → postprocess_running → pipeline_passed`记录成功路径，并分别使用`model_failed/postprocess_failed`区分失败阶段。`POSTPROCESS_WORKERS`可独立调整，但增加前应监测内存和许可证占用。

### 5.3 FSAF矩阵中的非有限数

Abaqus所带Python/NumPy在Windows下写CSV时，NaN可能表现为`-nan(ind)`、`nan(ind)`或`1.#QNAN`，普通`float()`不能解析其中部分形式。`Postprocess_All_surface_v2.py`现将这些标记统一恢复为NaN，再通过`valid_mask`排除无效频点；未知文本仍使矩阵读取失败，不能被静默替换为零或小量。该兼容只补全统一`s-grid`的`FSAF_1D_h/FSAF_station_h`派生表，不改变复数频响、PGA、RSAF或有效频带门槛。

### 5.4 G0—G1b历史工况清理记录

2026-07-22确认`Run/ch4_G0_config_injection`、`Run/ch4_G1_frequency_gate`和`Run/ch4_G1b_frequency_gate`仍分别承担配置注入证据、既有48次局部频带诊断证据和正式宽频闸门证据，因此三个目录均保留。完成关键清单、NPZ、Excel、日志和验收报告存在性核验后，永久删除其中248个可再生Abaqus过程文件，共约24.50 GiB；范围包括`.odb/.inp/.msg/.prt/.dat/.sta/.sim/.jnl/.com/.cae/.rpy`、`abaqus.rpy.*`以及实际存在时的恢复、锁定和求解中间后缀。清理后目标过程文件复核数为0；G0保留86个JSON和66份日志，既有G1保留26个NPZ、13个Excel、44个JSON和142份日志，G1b保留10个NPZ、10个Excel、38个JSON和54份日志。上述过程文件未进入回收站，只能通过外部备份恢复。

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
