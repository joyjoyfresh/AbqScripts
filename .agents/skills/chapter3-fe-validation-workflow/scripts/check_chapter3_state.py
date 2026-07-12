# -*- coding: utf-8 -*-
"""只读检查第三章执行状态、固定脚本和当前门禁。"""

from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import sys


STATE_RELATIVE = os.path.join('docs', '计划文档', '第三章执行状态.json')


def load_state(root):
    """读取并校验状态 JSON 的基础结构。"""
    path = os.path.join(root, STATE_RELATIVE)
    if not os.path.isfile(path):
        raise RuntimeError('missing state file: %s' % path)
    with open(path, 'rb') as fh:
        raw = fh.read()
    state = json.loads(raw.decode('utf-8'))
    for key in ('schema_version', 'canonical_files', 'fixed_pipeline', 'active', 'work_packages'):
        if key not in state:
            raise RuntimeError('state missing key: %s' % key)
    for key in ('phase_id', 'task_id', 'status', 'gate', 'solver_authorized'):
        if key not in state['active']:
            raise RuntimeError('active state missing key: %s' % key)
    return path, state


def check_paths(root, state):
    """检查规范文档和固定脚本是否存在。"""
    missing = []
    for _name, relative in sorted(state['canonical_files'].items()):
        if not os.path.isfile(os.path.join(root, relative.replace('/', os.sep))):
            missing.append(relative)
    for relative in state['fixed_pipeline']:
        if not os.path.isfile(os.path.join(root, relative.replace('/', os.sep))):
            missing.append(relative)
    return missing


def find_locks(root, state):
    """列出 Run 与 test/Abaqus 中的锁，并区分已知历史残留。"""
    locks = []
    for relative_root in ('Run', os.path.join('test', 'Abaqus')):
        scan_root = os.path.join(root, relative_root)
        if not os.path.isdir(scan_root):
            continue
        for directory, _subdirs, files in os.walk(scan_root):
            for name in files:
                if name.lower().endswith('.lck'):
                    relative = os.path.relpath(os.path.join(directory, name), root).replace(os.sep, '/')
                    locks.append(relative)
    known = set(path.replace('\\', '/') for path in state.get('known_stale_lock_files', []))
    all_locks = sorted(set(locks))
    known_locks = [path for path in all_locks if path in known]
    unknown_locks = [path for path in all_locks if path not in known]
    return all_locks, known_locks, unknown_locks


def find_abaqus_processes():
    """在 Windows 上列出实际 Abaqus 求解/前处理进程，许可证服务不计。"""
    if os.name != 'nt':
        return []
    try:
        process = subprocess.Popen(['tasklist', '/FO', 'CSV', '/NH'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _stderr = process.communicate()
        text = stdout.decode('mbcs', 'replace')
    except Exception:
        return []
    active = []
    exact = set(['standard.exe', 'explicit.exe', 'pre.exe', 'package.exe', 'abaqus.exe'])
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        name = row[0].strip().lower()
        if name == 'abaquslm.exe':
            continue
        if name in exact or (name.startswith('abq') and name.endswith('.exe')):
            active.append(row[0].strip())
    return sorted(set(active))


def main():
    """输出当前门禁摘要；发现结构错误时返回非零退出码。"""
    parser = argparse.ArgumentParser(description='检查第三章执行状态和固定入口')
    parser.add_argument('--root', required=True, help='AbqScripts 仓库绝对路径')
    parser.add_argument('--require-execution', action='store_true', help='要求研究者未暂停第三章任务执行')
    parser.add_argument('--require-solver', action='store_true', help='要求当前状态已授权求解')
    parser.add_argument('--json', action='store_true', help='以 JSON 输出检查结果')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isfile(os.path.join(root, 'AGENTS.md')):
        print('ERROR: invalid repository root: %s' % root, file=sys.stderr)
        return 2

    try:
        state_path, state = load_state(root)
        missing = check_paths(root, state)
    except Exception as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 2

    all_locks, known_locks, unknown_locks = find_locks(root, state)
    abaqus_processes = find_abaqus_processes()
    execution_hold = state.get('execution_hold', {})
    execution_hold_enabled = bool(execution_hold.get('enabled', False))
    result = {
        'state_file': os.path.relpath(state_path, root).replace(os.sep, '/'),
        'schema_version': state['schema_version'],
        'phase_id': state['active']['phase_id'],
        'task_id': state['active']['task_id'],
        'status': state['active']['status'],
        'gate': state['active']['gate'],
        'solver_authorized': bool(state['active']['solver_authorized']),
        'execution_hold_enabled': execution_hold_enabled,
        'execution_hold_reason': execution_hold.get('reason'),
        'execution_hold_lift_condition': execution_hold.get('lift_condition'),
        'next_task_after_gate': state['active'].get('next_task_after_gate'),
        'missing_files': missing,
        'lock_files': all_locks,
        'known_stale_lock_files': known_locks,
        'unknown_lock_files': unknown_locks,
        'abaqus_processes': abaqus_processes,
        'researcher_plan_approval': state.get('researcher_approval', {}).get('restructured_plan')
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for key in ('phase_id', 'task_id', 'status', 'gate', 'solver_authorized',
                    'execution_hold_enabled', 'execution_hold_reason', 'next_task_after_gate'):
            print('%s=%s' % (key.upper(), result[key]))
        print('MISSING_FILES=%d' % len(missing))
        print('LOCK_FILES=%d' % len(all_locks))
        print('KNOWN_STALE_LOCK_FILES=%d' % len(known_locks))
        print('UNKNOWN_LOCK_FILES=%d' % len(unknown_locks))
        print('ABAQUS_PROCESSES=%d' % len(abaqus_processes))
        for path in missing:
            print('MISSING: %s' % path)
        for path in known_locks:
            print('KNOWN_STALE_LOCK: %s' % path)
        for path in unknown_locks:
            print('UNKNOWN_LOCK: %s' % path)
        for name in abaqus_processes:
            print('ABAQUS_PROCESS: %s' % name)

    if missing:
        return 2
    if execution_hold_enabled and result['solver_authorized']:
        print('ERROR: execution hold requires solver_authorized=false', file=sys.stderr)
        return 2
    if execution_hold_enabled and state['active']['status'] == 'in_progress':
        print('ERROR: execution hold forbids active.status=in_progress', file=sys.stderr)
        return 2
    if (args.require_execution or args.require_solver) and execution_hold_enabled:
        print('ERROR: chapter 3 execution is paused by the researcher', file=sys.stderr)
        return 5
    if args.require_solver and not result['solver_authorized']:
        print('ERROR: solver is not authorized by the current gate', file=sys.stderr)
        return 3
    if args.require_solver and (unknown_locks or abaqus_processes):
        print('ERROR: active or unknown Abaqus lock/process detected', file=sys.stderr)
        return 4
    return 0


if __name__ == '__main__':
    sys.exit(main())
