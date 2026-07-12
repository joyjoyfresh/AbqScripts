---
name: chapter3-fe-validation-workflow
description: Continue, audit, or hand off this repository's thesis Chapter 3 finite-element credibility workflow. Use whenever a request mentions 第三章, 有限元验证, Hybrid v2, Autorun validation cases, Run/ch3 outputs, Shen 2024/2025 benchmarks, convergence/reflection checks, or writing validated Chapter 3 results into the thesis DOCX. The skill enforces plan/state startup checks, solver gates, evidence preservation, NPZ-based QA, rerun discipline, figure/table generation, Word rendering, and progress handoff.
---

# 第三章有限元验证工作流

## 目标

按仓库内唯一计划和执行状态，从当前门禁继续第三章工作。每个任务必须完成“预登记—Autorun—Abaqus—NPZ 检查—多门槛判定—诊断/回归—图表—Word 写入—状态更新”闭环，禁止把功能通过、单例诊断或求解器正常退出当作方法有效性。

## 启动时必须先做

1. 从当前目录向上定位包含 `AGENTS.md` 的仓库根目录。
2. 完整读取根 `AGENTS.md`。
3. 完整读取：
   - `docs/计划文档/第三章执行状态.json`
   - `docs/计划文档/第三章数值模型与可信性验证实施计划.md`
   - `docs/技术文档/第三章严格审查与重构论证.md`
4. 读取 `docs/技术文档/第三章数值模型验证记录.md` 中与当前任务有关的部分；若当前任务可能受旧缺陷影响，再读完整记录。
5. 运行：

   ```powershell
   python .agents/skills/chapter3-fe-validation-workflow/scripts/check_chapter3_state.py --root <仓库绝对路径>
   ```

   若准备执行任何建模脚本修改、Autorun、后处理改造或验证任务，必须再运行：

   ```powershell
   python .agents/skills/chapter3-fe-validation-workflow/scripts/check_chapter3_state.py --root <仓库绝对路径> --require-execution
   ```

6. 检查 `git status --short`，保护用户和其他 Agent 的未提交改动。
7. 检查是否已有 Abaqus/Autorun 进程或工况锁。禁止启动重复求解。

状态检查脚本会同时扫描 `Run/` 和 `test/Abaqus/`。已查明的历史残留锁会单列为 `known_stale_lock_files`；出现任何未知锁或 Abaqus 求解进程时，不得启动新求解。许可证服务 `ABAQUSLM` 不等于正在求解。

如果 `active.solver_authorized=false`，只能做当前门禁允许的审查、文档、静态检查或基础设施工作，不得启动求解器。计划、状态和用户最新指令冲突时，以用户最新指令为最高优先级，但必须同步更新计划和状态后再执行。

如果 `execution_hold.enabled=true`，表示研究者已暂停第三章执行。此时除读取/校验 Skill、计划、状态和修复这些工作流文件本身外，不得修改建模或后处理脚本、不得编写 Autorun、不得创建工况、不得启动求解，也不得把 `active` 标记为 `in_progress`。只有研究者在后续消息中明确要求继续，并同步解除状态文件中的暂停标志后，才允许进入下一任务。

## 固定文件和目录

论文验证固定使用：

- `Modeling/Hybrid/slope_frame_ssi_full_v2.py`
- `Postprocess/Hybrid/Postprocess_All_surface_v2.py`
- `Postprocess/Hybrid/Collect_All_results_v2.py`
- `Postprocess/Hybrid/Plot_Hybrid_surface_v2.py`

目录规则：

- 论文 Autorun：`Batch/Autorun_ch3_*.py`
- 论文 Abaqus 产物：`Run/ch3_*/run-###/`
- 临时诊断：`test/Abaqus/`
- 活动论文：`docs/论文材料/边坡地震动放大效应研究论文初稿（第三章重构）.docx`
- 原始初稿、整合修订稿和归档 Word：只读保留

收集和绘图必须直接消费 `surface_results.npz`；CSV 仅允许作为单工况后处理的运行期临时文件，不得重新建立“先长期输出 CSV 再收集”的数据链。

## 选择下一任务

只执行 `第三章执行状态.json` 中当前门禁允许的任务：

1. 若 `execution_hold.enabled=true`，立即停止任务执行，只报告已暂停和唯一待办；不得用“静态工作”绕过暂停。
2. 若 `active.status=in_progress`，先完成或明确冻结该任务，不得跳到后续任务。
3. 若当前任务失败，先保留失败 run 和报告，再诊断；不得通过换目录掩盖失败。
4. 若当前任务完成但状态未更新，先核验证据，再更新状态；不得重复求解。
5. 若状态为等待研究者决策，只完成不依赖该决策且未被暂停门禁禁止的静态工作，然后停止并清楚说明决策点。

正式顺序为 P0 → F0 → V1 → V2 → V3 → V4 → V5 → V6 → V7 → V8 → V9。具体工况数、指标和依赖以总计划为准，不在 Skill 中复制一套可能漂移的矩阵。

## 运行前预登记

在编写或启动 Autorun 前，把以下内容写入当前任务记录或工况清单：

- 研究问题和该验证不能被哪类证据替代；
- 工况选择原则、工况数和参数边界；
- 参考解来源及独立性；
- 比较变量、空间/时间/频率窗口、归一化分母；
- 主指标、辅助指标和事前通过阈值；
- 预期输出、最大运行时间、失败停止条件；
- 受影响的回归哨兵工况。

禁止看到目标曲线后才选择误差窗、删除异常点或放宽阈值。若必须改阈值，先停止，写明误差预算和理由，再让研究者确认。

## 编写和检查 Autorun

Autorun 应当：

1. 用 `sys.argv[1]` 接收工况根目录，默认落到 `Run/` 的对应任务目录。
2. 自动创建新的 `run-###`，不覆盖既有 run。
3. 以 `case_config.json` 注入参数，不用正则批量改生产脚本常量。
4. 复制或记录四个固定脚本的哈希，记录输入波哈希和环境信息。
5. 串行执行建模、求解、单工况后处理、NPZ 收集和统一绘图。
6. 对每一步检查退出码、必需文件和日志关键状态；未完成 ODB 不得后处理。
7. 失败时写 `status=failed/invalid`，保留原日志、ODB/残片和配置。
8. 对实际 `dt`、网格、单元、阻尼、域尺寸和输入波做结果侧核验，不能只相信注入配置。

移动或新建脚本后执行 UTF-8 `py_compile`、绝对路径解析和模拟 `__file__` 检查。Abaqus Python 2.7 代码不得使用只在 Python 3 可用的语法。

## 求解后的强制检查

求解器退出码为 0 仅代表流程条件之一。至少检查：

- `.sta/.msg/.dat/.log` 是否表明正常完成；
- ODB 是否完整、步和帧数是否符合预期；
- `case_meta.json` 和 `case_config.json` 是否与预登记一致；
- `surface_results.npz` 的 manifest、时程、元数据和 QA 字段是否齐全；
- 实际 dt、单元数、网格尺寸、单元类型、阻尼和域尺寸；
- 参考解是否来自独立实现，是否误调生产 FD 内核；
- 原始曲线、误差曲线和负对照是否能检出故意错误；
- 所有强制 QA 是否同时通过。

`overall_pass` 必须是强制门槛的逻辑合取。`qa_window_convergence` 不得覆盖 `qa_theory` 或 `qa_reflection` 的失败。详细门槛见 `references/validation-gates.md` 和总计划。

## 失败、修改和回归

失败时按以下顺序处理：

1. 冻结 run，生成失败摘要，记录最早异常证据。
2. 区分配置/路径、建模、求解、后处理和物理不一致。
3. 提出可证伪的根因假设，只做最小修改。
4. 优先用已有 ODB 重提取，只有模型/求解改变时才重算。
5. 修改主建模脚本、后处理协议或 QA 逻辑后，至少回归均质解析、成层参考和控制坡地各一个哨兵。
6. 回归全部通过后才能重跑目标矩阵。

不得用增加阻尼、缩短时窗、删除下游数据或放宽阈值来获得“通过”。同一物理偏离经两轮有证据的修复仍不能解释时，停止并形成方法学决策项。

## 图表和论文写入

只有对应任务全部通过后才允许写入论文：

1. 保存原始曲线、误差曲线、负对照和指标表；平滑图不能替代原始图。
2. 每一小节按“目的—必要性—工况—参数—指标/阈值—结果—问题解释—支撑范围”写成连续学术段落。
3. 不使用“吻合良好”“证明完全正确”等无指标表述。
4. 在活动工作稿中写入，不修改原始初稿、整合修订稿和归档文件。
5. 更新 Word 域和目录，导出 PDF，用 Poppler 渲染相关页检查分页、表格、图题、空白页和字体。
6. 图、表、数据和正文中的工况编号必须能回指到 `Run/`、Autorun 和 NPZ。

## 状态更新和交接

按“证据先落盘，状态后更新”的顺序：

1. 写验收/失败报告和图表。
2. 更新 `第三章数值模型验证记录.md`。
3. 更新总计划的进度表和必要的方案调整。
4. 最后更新 `第三章执行状态.json` 的当前任务、状态、门禁和下一任务。
5. 重新运行状态检查脚本，确认 JSON、固定文件和路径有效。

交接说明至少包含：本轮任务、运行目录、通过/失败门槛、修改文件、回归结果、论文写入位置、尚存风险和唯一下一任务。不得仅写“继续下一步”。

## 禁止事项

- 不得把 U0—U3b 历史状态恢复为正式通过，除非总计划明确重新验证。
- 不得使用旧正文的 522 组复现、远场小于 2%、域尺寸扰动或反射率结论。
- 不得把共享自由场/边界算法的 OpenSees 对比称为完全独立验证。
- 不得在未安装/验证运行时前假定 SPECFEM2D 或 OpenSees 可用。
- 不得在论文生产 EQL 工况中允许异常回退线性。
- 不得清理失败 run、原始日志、旧 Word 或用户未提交改动。

## 相关参考

- `references/validation-gates.md`：证据等级和 QA 合取规则。
- `references/state-schema.md`：执行状态字段及更新约束。
- 仓库总计划：完整工况矩阵、章节结构和写入顺序的唯一来源。
