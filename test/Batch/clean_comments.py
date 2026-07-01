# -*- coding: utf-8 -*-
"""
清洗 VAB_oblique_multilayer_nonlinear_v2.py 的中文注释：
剔除"复述代码在做什么"的平凡行内注释，保留关键注释。

保留规则：
  - 段落标题 (# ====... / # -----...)
  - docstring 行
  - 含关键意图词的注释（注意/防止/避免/兼容/因为/否则/退化/口径/用于/论文/约定/阈值/临界/公式/vN 标记等）
  - 较长、含物理/设计含义的注释
剔除规则：
  - 短小的动作复述（初始化/遍历/计算/返回/设置/读取/获取/累加/拼接/记录日志 等）
"""
import re
import io

SRC = "../Modeling/Multi/VAB_oblique_multilayer_nonlinear_v2.py"
DST = "../Modeling/Multi/VAB_oblique_multilayer_nonlinear_v2.py"

# 关键意图词：命中即保留
KEEP_KW = [
    '注意', '防止', '避免', '兼容', '因为', '否则', '退化为', '退化', '口径',
    '用于', '供建材', '保证', '论文', '约定', '阈值', '临界', '公式', '原理',
    '要点', '关键', '机制', '校验', '校核', '对齐', '同源', '同口径', '不破坏',
    '不重叠', '不留缝', '不应', '不会', '不可', '必须', '务必', '确保', '保证',
    '详见', '见下', '参见', '对应', '即', '等于', '为:', '为：',
    'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
    'Py2', 'Py3', 'Abaqus', 'SHAKE', 'EQL', 'TAF', 'K-L', 'Kuhlemeyer',
    'Snell', 'Ricker', '瑞利', '阻尼比', '品质因子', '半空间', '自由面',
    '硬化', '拦截', '中止', '回退', '兜底', '降频', '瘦身', '延长', '截断',
    '覆写', '注入', '覆盖默认', '默认', '强制', '钳位', '截断到',
]

# 平凡动作前缀：仅由这些动词开头且无关键意图词 → 剔除
TRIVIAL_PREFIX = [
    '初始化', '遍历', '计算', '返回', '设置', '读取', '获取', '取', '由',
    '记录', '输出', '继续', '跳过', '清空', '累加', '更新', '追加', '保存',
    '覆盖', '创建', '定义', '构造', '组装', '生成', '提取', '换算', '规范化',
    '收集', '检查', '统计', '拼接', '转换', '保存', '复制', '记录', '提交',
    '等待', '删除', '打包', '填入', '填入', '加入', '判断是否', '判断',
    '安全获取', '安全读取', '递归', '按', '从', '对', '为', '用', '将', '把',
    '返回', '返回计算', '返回该', '返回该带', '返回可用', '返回总', '返回基',
    '设置该', '设置当前', '设置法', '设置切', '设置区', '设置弹', '设置阻',
    '设置网格', '设置输出', '设置分析', '设置作业', '设置模型', '设置内存',
    '设置时间', '设置默认', '设置日志', '设置统一', '设置单元',
    # 第二轮补充：仍是动作复述的叙述注释（保留变量含义标签，不删）
    '说明', '构建', '绑定', '添加', '处理', '使用', '结束', '其余', '其他',
    '归入', '标记为', '继续下一', '继续', '兜底返回', '兜底', '顺序提交',
    '重新生成', '需要补零时', '有日志器时', '未命中', '命中',
    '默认使用', '默认分析步', '非最底层须有', '非法边界', '阶段标记',
]


def split_code_comment(line):
    """返回 (code, sep, comment) ；无注释时 sep/comment 为 ''。
    粗略跳过字符串内的 #（单/双引号）。"""
    code = []
    i = 0
    n = len(line)
    in_s = None  # 当前字符串引号
    while i < n:
        ch = line[i]
        if in_s:
            code.append(ch)
            if ch == '\\':  # 转义
                if i + 1 < n:
                    code.append(line[i + 1])
                    i += 2
                    continue
            elif ch == in_s:
                in_s = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_s = ch
            code.append(ch)
            i += 1
            continue
        if ch == '#':
            return ''.join(code), '#', line[i + 1:]
        code.append(ch)
        i += 1
    return ''.join(code), '', ''


def is_trivial(comment_text):
    """判断注释正文是否为平凡复述（应剔除）。"""
    t = comment_text.strip()
    if not t:
        return False  # 空 # 保留与否由调用方处理（这里返回 False 表示不动）
    # 含关键意图词 → 保留
    for kw in KEEP_KW:
        if kw in t:
            return False
    # 纯版本标记/结束标记行内的也保留已由 kw 命中
    # 检查是否以平凡动词开头
    for pf in TRIVIAL_PREFIX:
        if t.startswith(pf):
            return True
    return False


def process(path):
    with io.open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    out = []
    removed = 0
    in_docstring = False  # 简单三引号跟踪
    for ln in lines:
        stripped = ln.lstrip()
        # docstring 跟踪（粗略）
        if '"""' in ln:
            cnt = ln.count('"""')
            if cnt % 2 == 1:  # 奇数个三引号：多行 docstring 开始/结束
                # 纯闭合三引号行（仅 """ + 可选尾注释）：剔除平凡尾注释
                m = re.match(r'^(\s*)"""(\s*#.*)?$', ln)
                if m and m.group(2):
                    ctext = m.group(2).strip().lstrip('#').strip()
                    if is_trivial(ctext):
                        out.append((m.group(1) + '"""').rstrip())
                        removed += 1
                        in_docstring = not in_docstring
                        continue
                in_docstring = not in_docstring
                out.append(ln)
                continue
            # 偶数个（含2）：本行自洽的单行 docstring → 继续走正常流程，剔除平凡尾注释
        if in_docstring:
            out.append(ln)
            continue

        # 段落标题行（整行注释）保留
        if stripped.startswith('#'):
            out.append(ln)
            continue

        code, sep, comment = split_code_comment(ln)
        if sep:
            if is_trivial(comment):
                # 剔除：去掉行内注释及前导空白
                new_line = code.rstrip()
                out.append(new_line)
                removed += 1
                continue
        out.append(ln)

    with io.open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print("removed inline trivial comments:", removed)
    print("total lines now:", len(out))


if __name__ == '__main__':
    process(SRC)
