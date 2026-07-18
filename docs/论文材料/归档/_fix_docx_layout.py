# -*- coding: utf-8 -*-
"""修复论文修订稿中表格行被跨页拆分造成的孤立文本。"""

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


DOCX_PATH = r"C:\Users\12462\Documents\Code\AbqScripts\docs\论文材料\边坡地震动放大效应研究论文初稿（整合修订）.docx"


def set_cant_split(row):
    """禁止一个表格行跨页拆分。"""
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def set_repeat_header(row):
    """将首行标记为跨页重复表头。"""
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:tblHeader")) is None:
        tr_pr.append(OxmlElement("w:tblHeader"))


def main():
    document = Document(DOCX_PATH)
    matched = 0
    for table in document.tables:
        text = "\n".join(cell.text for row in table.rows for cell in row.cells)
        if "坡顶平台长度 L_A" not in text:
            continue
        for row in table.rows:
            set_cant_split(row)
        set_repeat_header(table.rows[0])
        matched += 1
    if matched != 1:
        raise RuntimeError("目标表格数量异常：%d" % matched)
    document.save(DOCX_PATH)
    print("已修复目标表格跨页拆分：%s" % DOCX_PATH)


if __name__ == "__main__":
    main()
