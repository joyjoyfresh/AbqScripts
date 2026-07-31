# Abaqus 批处理运行踩坑记录

## 1. 背景

在执行 P1 左右观测窗外扩诊断时，曾临时创建 `Batch/Autorun_P1_window_test_v1.py` 并尝试通过后台 `Start-Process` 与 Codex `shell_command` 启动 Abaqus 批处理。该过程暴露出两个流程问题：

1. 临时诊断性质的Abaqus批处理脚本不应放在`Batch/`，应放在`test/Batch/`；`Batch/`保留通用模板，论文正文研究单元的正式入口统一放在`Run/Auto_ch4/`；
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

与论文正文相关的正式研究Autorun统一放在`Run/Auto_ch4/`，其Abaqus工况与结果统一放在`Run/<研究单元>/run-###/`；`Batch/Autorun_template_v2.py`只作为派生模板，不直接承担具体正文数据池。脚本默认自动递增运行编号，避免覆盖既有证据。临时诊断脚本若需输出到`Run/`下，应通过命令行参数显式指定，例如：

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
[运行状态][工况 3/51][case-H1-000003] 作业进度: job-example，已算到 12.345 秒/共 45.900 秒（26.9%，Step-earthquake）
```

其中`3/51`表示该工况在冻结工况清单中的计划序号和本批总数，不表示完成次序；并发运行时多个序号可能交错输出。建模开始、脚本开始/完成、单工况后处理和定期作业进度均使用同一前缀。多工况并发时每条消息使用终端输出锁整行打印，避免字符交叉。Autorun只选择启动时间之后新建或更新的`.sta`，不会误报同目录上一次运行的状态；若`.sta`尚未产生或总时长暂未识别，才回退转发建模日志中的`作业进度`行。建模日志默认每6分钟更新，终端默认每5分钟直接采样求解器状态，两者互不依赖。子进程完成状态另以5秒间隔轻量检查，检测到结束后立即读取最终进度，不会因低频进度显示而推迟后处理或下一个工况。`.sta`可在工况通过数据提取门后按清理规则删除，但运行过程中不得提前删除。

后处理日志在Abaqus Python 2.7中必须先把中文步骤名和消息统一格式化为Unicode，再交给日志处理器编码为UTF-8。旧写法让字节串与Unicode参数在`logging`内部触发隐式ASCII解码，可能出现`UnicodeDecodeError`，但异常只发生在日志输出层而不一定中止数据提取。`Postprocess_All_surface_v2.py`已统一使用Unicode安全格式化，并由普通Python单元测试和Abaqus Python 2.7实际写日志冒烟测试覆盖；后续不得以删除中文日志绕过编码问题。

G1r专用入口`Run/Auto_ch4/Autorun_ch4_G1r.py`采用相同的`.sta`直读逻辑，并已完成8个正式真实波工况。源码修改不会热加载到已经启动的Autorun或已复制到运行目录的建模脚本，正在运行的批次保持原行为且无须中断。G1r入口支持用`--release-evidence-root`和`--initial-diagnostic-root`分别导入组合宽频门与增量初态诊断门，并同时锁定门控文件、门控内容和源运行清单的SHA-256。执行`--run-real-only`时会再次核验这些证据，任何源证据变化都会阻止启动Abaqus。该批采用单机4个建模槽和1个后处理槽稳定完成，因此通用模板的后续默认并发冻结为相同设置，单作业CPU和内存设置不变。

若旧版控制逻辑在求解成功后停留于100%轮询，只有在`.sta`明确出现成功完成标志、`.msg`无错误且`.lck`消失后，才允许终止残留Autorun/CAE外层进程并对既有ODB单独后处理；不得把仍在增长的ODB当作已完成结果。恢复后应保留独立审计文件，记录实际执行脚本、ODB、NPZ和Excel的SHA-256。G1r运行清单还会按工况写入`case_source_file_sha256`，用于区分恢复前后仅控制逻辑不同的脚本，避免用一个全局源码哈希掩盖批次内差异。

### 5.2 建模—单工况后处理流水线

通用模板和G1r正式入口将每个工况的本地流程拆成两个独立队列：`MODEL_SCRIPT_SEQUENCE`由建模线程池领取，`CASE_POSTPROCESS_SCRIPT_SEQUENCE`由单工况后处理线程池领取。某工况建模进程正常退出后，状态先写为`model_passed`并立即提交后处理；原建模槽同时领取下一个`planned`工况，不再等待前一工况的ODB提取完成。全局汇总和跨工况绘图仍在全部单工况后处理结束后执行，不能提前读取半成品。

当前正式配置为4个建模/求解槽和1个单工况后处理槽，因此最多同时存在4个求解流程与1个只读ODB后处理流程，但求解作业并发上限仍是4。建模使用`abaqus cae noGUI=...`，后处理按脚本约定使用更轻量的`abaqus python ...`。若建模失败，不提交对应后处理；若后处理失败，保留ODB和日志。清单按`model_running → model_passed → postprocess_running → pipeline_passed`记录成功路径，并分别使用`model_failed/postprocess_failed`区分失败阶段。`POSTPROCESS_WORKERS`可独立调整，但增加前应监测内存和许可证占用。

### 5.3 数据提取后的自动清理

通用模板在单工况后处理结束后立即检查`surface_results.npz`和`surface_results.xlsx`是否同时存在且非空。后处理v2会在两项规范结果写出后再次读取`surface_summary.json`，并额外写出`postprocess_status.json`冻结逐记录必需QA状态。外层批处理不再只依赖`abaqus.bat python`的进程退出码，而会读取该状态文件；缺文件、无法解析或`passed=false`均判为`postprocess_failed`。只有必需QA通过、轻量状态通过且两项规范产物齐全时，才删除`.odb/.inp/.msg/.prt/.dat/.sta/.sim/.jnl/.com/.rpy/.rec`；任一步失败或任一规范产物缺失时均跳过清理，保留完整现场用于诊断。

当前存储策略明确保留`.cae`，同时保留配置、元数据、输入、脚本、日志、质量JSON、NPZ、Excel和图表。每次成功清理都在工况目录写入`cleanup_audit.json`，记录规范产物的大小与SHA-256、删除文件名、删除失败项和实际释放字节数。清理是永久删除，不进入回收站；若后续需要从ODB提取新增变量，只能重新求解，因此正式后处理字段必须在批跑前冻结。

G1r正式`run-003`已按该口径完成删除前散列冻结和清理：84个过程文件被永久删除，释放`29.795 GiB`，目标过程文件残留数为0；8个CAE、8个NPZ、8个Excel和8份清理审计保留。根目录的`G1r正式冻结清单.json`另外记录删除前ODB散列、恢复前后脚本变体和科学门文件散列。

H1—H3正式入口`Run/Auto_ch4/Autorun_ch4_H_v1.py`已继承上述进度、流水线和清理规则。`run-001`的`H1-000035/H2-000006/H3-000006`已使用3槽完成，0.5—10 Hz覆盖、尾段、能量、初态和结果散列均正常，但`side_clearance=0.1h`没有既有收敛证据，传播上游端点相对一维误差达到23.83%—53.69%；统一复频响网格还在坡脚缺1—2个点。因此`run-001/H1_H3预生产放行门.json`冻结为`passed=false`，剩余48个工况不得继续。

`run-002`把侧向净空改为`1.0h`，能量加入所有正式工况必需门，3个首批工况另外要求计算域门；统一复频响只对物理连续的地表场补齐几何棱附近短缺口。3个工况均完成求解和规范结果提取，当前水平/垂向复频响在801个空间点全部有效，频带、尾段和能量也通过；但相对`run-001`的全窗口最大同点差为17.71%/4.22%/17.97%，H1/H3复频响幅相超限，因此`run-002/H1_H3预生产放行门.json`冻结为`passed=false`。

本轮还发现Windows下`abaqus.bat python`没有可靠传回`sys.exit(3)`，导致NPZ内部`overall_pass=false`却被外层清单误记为`pipeline_passed`，并提前删除了过程文件。后处理与通用模板已按本节开头改用`postprocess_status.json`闭环；该修复只作用于后续新run，不能恢复已经删除的ODB。

H入口随后把参考场从旧`local_columns`修正为三边界统一`global_upper`，最终预生产目录为`run-005`。首批配置引用`run-002`的旧结果属于QA对照错误：它同时改变了参考场算法、坡高和坐标体系，不满足计算域收敛的单变量原则。H1/H2/H3均已完成求解，3个工况的FRF和能量QA实际通过；旧`required_qa_failed`不代表模型或求解失败。当前脚本已取消正式H工况之间的必需`domain`比较，3个现有ODB用当前后处理重新提取并重分类为`pipeline_passed`，没有重新调用求解器。原配置、脚本、状态和日志均保留`旧域门备份`，根目录另有`H1_H3旧域门重分类审计.json`。

坡地传播上游端点的总响应包含坡体反射与表面波散射，也不再强制与无坡一维PGA相等；一维端点闭环只在`validation_geometry='flat'`平场验证中作为理论门。`side_clearance`是坡顶`4H`和坡脚`3H`观测窗以外的单侧附加净空，`side_clearance=1H`时边界实际已距坡肩`5H`、距坡脚`4H`。旧`0.1H→1H`差异只能证明`0.1H`不足，不能证明`1H`不足，因而取消未启动的`4H→6H`扩域方案。

H坡地正式域门改为经济候选域与大域参考的同口径最不利对照：新建`HD-H3-000001 global_upper 1H`，对比现有`run-005/H3-000006 global_upper 4H`，坡高、坡角、入射角、材料、网格、输入、时间窗和后处理全部固定。专用入口和正式目录分别为`Run/Auto_ch4/Autorun_ch4_H_domain_v1.py`与`Run/ch4_H_domain_convergence/run-001`，准备阶段不启动Abaqus，运行命令为：

```powershell
python "Run\Auto_ch4\Autorun_ch4_H_domain_v1.py" "Run\ch4_H_domain_convergence\run-001" --run
```

该入口完成后自动比较统一`s∈[-4,4]`窗口和`0.5—10 Hz`频带的`log|H|` RMSE、相位圆周RMSE、两者P95、共同有效覆盖率及`TAF_h_comp`相对差。阈值已在求解前写入`H计算域收敛预注册.json`并冻结。`run-001`实际结果中覆盖率、相位和`TAF_h_comp`通过，但`log|H|` RMSE=`0.09959>0.05`、幅值对数差P95=`0.16422>0.10`，故1H未通过。

考虑到2H预计约`100264`个单元，比4H的`135264`个单元少约`25.9%`，且后续尚有48个正式均质工况，追加一次2H验证约在3个正式工况内即可回收成本。最终候选目录为`Run/ch4_H_domain_convergence/run-002`，仍只改变侧向净空并复用相同4H参考、输入、脚本和7项门槛。运行命令为：

```powershell
python "Run\Auto_ch4\Autorun_ch4_H_domain_v1.py" "Run\ch4_H_domain_convergence\run-002" --run
```

该目录的预注册协议同时冻结决策：2H全项通过则后续正式生产采用2H；任一项失败则采用4H；不放宽门槛，也不再追加其他计算域尺寸。实际2H对4H的覆盖率为100%，相位RMSE/P95=`0.09180/0.12810 rad`，TAF差P95/最大=`3.168%/5.129%`，均通过；但`log|H|` RMSE=`0.08822>0.05`、幅值对数差P95=`0.13004>0.10`，因此按当时“完整表面原始FRF全项通过”的单一口径，2H未通过并冻结4H。后续2H限定输出决策不改写该历史结果。

`run-005/H1_H3预生产放行门.json`已锁定1H/2H域门文件散列、4H生产域、3个代表工况配置散列及NPZ/Excel散列。放行门生成前再次核验51份配置，状态为3个`pipeline_passed`和48个`planned`。代表工况删除前还冻结ODB、STA和CAE散列；随后按自动清理规则永久删除30个过程文件，共释放`11.638 GiB`，3个CAE、NPZ、Excel、质量JSON和逐工况`cleanup_audit.json`保留。该门在形成时允许以下4H续跑命令，但2026-07-28用户改选2H限定输出方案后，本命令暂停，不再执行：

```powershell
python "Run\Auto_ch4\Autorun_ch4_H_v1.py" "Run\ch4_H_homogeneous_baseline\run-005" --run-remaining
```

旧入口会再次核验4H域宽、两个域门文件散列、3个代表工况流程状态以及NPZ/Excel散列；任一证据漂移都在调用Abaqus前停止。该历史门和`run-005`状态保留，不改写为2H。

2026-07-28用户最终决定接受2H的已知净空偏差。历史1H/2H/4H诊断和完整表面原始FRF失败标签保持不变，但J0英文期刊支线不再设置任何新增2H/4H放行门。旧H `run-006`保留为51/51 `planned`的历史准备目录，不再执行其中的配对或剩余工况；J0在独立目录重新包含51个均质设计，避免跨run复用旧门、旧状态或不同来源散列。

J0正式入口为`Run/Auto_ch4/Autorun_ch4_J0_v1.py`。它一次生成175个统一`side_clearance=2.0h`目录，其中JL2B的16个目录为学习曲线条件扩样；159个基础目录对应163次动力求解，条件上限为179次。准备命令不调用Abaqus：

```powershell
python "Run\Auto_ch4\Autorun_ch4_J0_v1.py" --prepare-only
```

2026-07-28已用最终入口生成有效目录`Run/ch4_J0_journal_one_month/run-002`。清单复核为175个目录、179行求解记录、0个非2H配置；状态为115个`planned`、16个`blocked_learning_curve`、28个`sealed_model_lock`、12个`blocked_model_lock`和4个`blocked_final_evaluation`。求解产物、身份审计和结果文件计数均为0，预生产清单及状态审计均记录`abaqus_started=false`。同级`run-001`生成于最终恢复锁补丁之前，虽然也未启动Abaqus，但来源散列已失效，必须保留为不可执行快照或由用户另行清理，不得用于正式求解。

入口不允许无参数默认开跑。`homogeneous`和`core`可以在准备审计通过后显式启动；`reserve`要求人工写入学习曲线门，`blind`和`ood`要求模型冻结门，`real_wave`还要求盲测与边界评价完成门。任何阶段都应先用`--dry-run`核对将领取的`planned`工况：

```powershell
python "Run\Auto_ch4\Autorun_ch4_J0_v1.py" "Run\ch4_J0_journal_one_month\run-002" --run-stage homogeneous --dry-run
python "Run\Auto_ch4\Autorun_ch4_J0_v1.py" "Run\ch4_J0_journal_one_month\run-002" --run-stage core --dry-run
```

J0还在建模成功与后处理之间增加`case_meta.json`身份核验。建模脚本遇到配置解析异常时可能回退到默认配置并继续退出为0，因此入口必须逐项比较入射角、几何、坡高、基岩材料、覆盖层材料与厚度、表面几何和验证几何；任何不一致都在`config_identity_audit.json`中标记`passed=false`，并把流程状态写为`model_failed`，禁止后处理和自动清理。这一门防止“软件运行成功但物理工况跑错”进入训练集。

外层进程返回0也不能替代逐记录求解核验。J0对每个预期`record_id`要求同名`job-<record>-slope.sta`包含`THE ANALYSIS HAS COMPLETED SUCCESSFULLY`、对应ODB存在且非空、MSG不存在显式`***ERROR`、LCK消失，并要求目录中的ODB/STA集合与配置记录集合完全一致；检查结果写入`solver_completion_audit.json`。后处理后还会复读`postprocess_status.json`，要求记录集合一一对应、每条`overall_pass=true`且NPZ/Excel属于当前执行；JW0还单独要求每条记录的水平绝对PSA数组存在且有限，结果统一写入`result_completeness_audit.json`。因此JW0双波目录缺任一条记录时不能整体标为`pipeline_passed`。

启动后续阶段时，入口同时复核阶段门绑定的预生产清单散列、采样冻结清单散列和前置工况实际状态。人工签署的模型门不能绕过尚未完成的开发工况，最终评价门也不能绕过未完成的JL3、JC0或JL5。

J0成功工况只删除`.odb/.inp/.prt/.sim/.jnl/.com/.rpy/.rec`，保留CAE、STA、MSG、DAT、日志、NPZ和Excel以支持长批次恢复审计；失败工况不清理。重复执行阶段只领取`planned`状态，失败状态不自动重试。状态审计命令只读结果与求解产物，不调用Abaqus：

```powershell
python "Run\Auto_ch4\Autorun_ch4_J0_v1.py" "Run\ch4_J0_journal_one_month\run-002" --audit-status
```

若外层进程在求解成功后中断，先用`--recover-stage <阶段> --dry-run`列出`model_running/model_passed/postprocess_running`工况，再显式执行同一恢复命令。恢复模式只在全部预期STA/MSG/ODB、身份和记录集合通过时重分类或重新后处理，绝不重新提交求解；证据写入`J0中断恢复审计.json`。`model_failed/postprocess_failed`不会自动领取，必须先诊断原因并另行决定。正式`run-stage`和`recover-stage`共用run级操作系统文件锁；已有进程持锁时第二个进程会在状态改写或后处理启动前停止。正常退出会显式释放并把所有者状态改为`released`；父进程异常退出时操作系统虽释放文件锁，但遗留的`active`所有者记录会阻止后续实际写入。只有人工确认旧Autorun及其`abaqus python`后处理均已退出后，才允许增加`--acknowledge-stale-owner`接管，旧所有者信息会写入新锁记录。单个工况的状态文件缺失、证据解析、STA复核或来源复制异常只记为`unresolved`并继续检查其余工况，不会中断整批恢复：

```powershell
python "Run\Auto_ch4\Autorun_ch4_J0_v1.py" "Run\ch4_J0_journal_one_month\run-002" --recover-stage core --dry-run
python "Run\Auto_ch4\Autorun_ch4_J0_v1.py" "Run\ch4_J0_journal_one_month\run-002" --recover-stage core
```

### 5.4 FSAF矩阵中的非有限数

Abaqus所带Python/NumPy在Windows下写CSV时，NaN可能表现为`-nan(ind)`、`nan(ind)`或`1.#QNAN`，普通`float()`不能解析其中部分形式。`Postprocess_All_surface_v2.py`现将这些标记统一恢复为NaN，再通过`valid_mask`排除无效频点；未知文本仍使矩阵读取失败，不能被静默替换为零或小量。对物理连续的`H_surface_h/v`、地表`FSAF`和绝对`PSA`，程序允许用两侧真实节点补齐不超过`Δs=0.15`的内部短缺口；端外、宽缺口、一维谱比、台站谱比和反应谱放大比均不补值，避免把真实口径跳变抹平。

### 5.5 G0—G1b历史工况清理记录

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
3. 通用批处理模板放入`Batch/`，正文数据池正式入口放入`Run/Auto_ch4/`；
4. 长时间 Abaqus 求解尽量由用户终端前台运行，Codex 负责准备脚本、检查日志和分析结果；
5. 判断完成必须以求解成功标志、规范NPZ/Excel、后处理日志和质量门为准；若ODB已按审计清理，则以冻结清单中的删除前散列和逐工况清理审计证明其曾存在并完成提取。
