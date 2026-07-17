from __future__ import print_function

import re
import shutil
import zipfile
from pathlib import Path

from docx import Document

from Export_thesis_chapters_markdown import (
    clean_text,
    heading_level,
    iter_body,
    markdown_table,
    paragraph_images,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs' / '论文材料' / '边坡地震动放大效应研究论文初稿.docx'
OUTPUT_DIR = ROOT / 'docs' / '论文材料' / '归档'
OUTPUT = OUTPUT_DIR / '第3章_坡地有限元模型建立与验证_v1.md'
ATTACHMENTS = OUTPUT_DIR / '附件_第3章_v1'
TARGET_HEADING = '3 坡地有限元模型建立与验证'


def extract_images():
    ATTACHMENTS.mkdir(parents=True, exist_ok=True)
    image_map = {}
    with zipfile.ZipFile(str(SOURCE), 'r') as archive:
        for name in archive.namelist():
            if not name.startswith('word/media/'):
                continue
            target = ATTACHMENTS / Path(name).name
            with archive.open(name) as source, target.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            image_map['/' + name] = target.name
    return image_map


def main():
    image_map = extract_images()
    doc = Document(str(SOURCE))
    blocks = []
    in_target = False

    for kind, item in iter_body(doc):
        if kind == 'paragraph':
            text = clean_text(item.text)
            level = heading_level(item)
            normalized = re.sub(r'\s+', ' ', text).strip()
            if level == 1 and normalized == TARGET_HEADING:
                in_target = True
                continue
            if level == 1 and in_target:
                break
            if not in_target:
                continue
            for image in paragraph_images(item):
                if image in image_map:
                    blocks.append(('image', image_map[image]))
            if not text:
                continue
            if level == 2:
                blocks.append(('text', '## ' + text))
            elif level == 3:
                blocks.append(('text', '### ' + text))
            elif text.startswith('【图片占位】'):
                blocks.append(('text', '> 📌 ' + text))
            else:
                blocks.append(('text', text))
        else:
            if in_target:
                table = markdown_table(item)
                if table:
                    blocks.append(('table', table))

    if not in_target:
        raise RuntimeError('未找到原始第三章标题：{}'.format(TARGET_HEADING))

    lines = [
        '# 第3章 坡地有限元模型建立与验证（v1 原始版）',
        '',
        '*从 `边坡地震动放大效应研究论文初稿.docx` 提取；本文件保留原始第三章内容，不包含后续第三章重构稿的改写。*',
        '',
        '---',
        '',
    ]
    for index, (kind, value) in enumerate(blocks):
        if kind == 'image':
            lines.append('![原始第三章图形附件](附件_第3章_v1/{})'.format(value))
            lines.append('*图形附件：从原始 Word 文档中提取。*')
        else:
            lines.append(value)
        if index != len(blocks) - 1:
            lines.append('')

    if not image_map:
        lines.extend([
            '',
            '> 📌 原始 Word 文档未嵌入实际图片；原始第三章中的图形均以“图片占位”文字保留，因此 `附件_第3章_v1/` 为空。',
        ])
        readme = ATTACHMENTS / 'README.md'
        readme.write_text('原始第三章 Word 文档未包含嵌入图片；图形仅以占位文字存在。\n', encoding='utf-8')

    OUTPUT.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    print('已导出原始第三章 v1：{}'.format(OUTPUT))
    print('图片附件数量：{}'.format(len(image_map)))


if __name__ == '__main__':
    main()
