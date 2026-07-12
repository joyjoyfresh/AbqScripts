# 第三章执行状态约束

状态文件：`docs/计划文档/第三章执行状态.json`。

## 关键字段

- `schema_version`：状态结构版本。
- `canonical_files`：计划、审查、记录、活动 Word 和归档路径。
- `fixed_pipeline`：当前唯一建模与后处理脚本链。
- `active.phase_id/task_id/status/gate`：当前唯一任务和门禁。
- `active.solver_authorized`：是否允许启动求解器。
- `active.next_task_after_gate`：当前门禁完成后的唯一下一任务。
- `execution_hold.enabled`：研究者是否暂停第三章任务执行；为 `true` 时优先级高于 `active`。
- `execution_hold.reason/set_at/lift_condition`：暂停原因、登记时间和解除条件，防止其他 Agent 误判。
- `researcher_approval`：计划是否已由研究者确认；F0 前必须显式为已批准。
- `historical_evidence`：U0—U3b 的降级状态，不能自动升级。
- `work_packages`：P0、F0、V1—V9 状态。
- `environment_findings`：Abaqus/OpenSees/SPECFEM2D 当前可用性，只能由实测更新。
- `known_stale_lock_files`：已查明无对应求解进程的历史残留锁，仅用于区分未知活动锁，不代表可以删除。

## 更新规则

1. 状态必须描述真实证据，不描述计划愿望。
2. 在报告、日志、NPZ、图件和验证记录落盘前，不得标记任务完成。
3. `solver_authorized` 只能根据用户授权和计划门禁设置；不能因“下一任务需要求解”自动改为 `true`。
4. `execution_hold.enabled=true` 时，`solver_authorized` 必须为 `false`，`active.status` 不得为 `in_progress`；只有研究者明确要求继续后才能解除。
5. 环境可用性必须由当前会话实测，不从旧记录推断。
6. 修改 JSON 后运行 `scripts/check_chapter3_state.py`；开始任何任务动作前运行同一脚本的 `--require-execution` 门禁。
7. 新增字段保持向后兼容；变更现有字段含义时提升 `schema_version` 并同步修改 Skill。
