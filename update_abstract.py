#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update abstract and keywords in the combined crisis paper docx."""

from docx import Document

from add_part3_abstract import ABSTRACT, KEYWORDS

DOCX = "/workspace/1997亚洲金融危机.docx"


def replace_para_text(para, new_text):
    if para.runs:
        para.runs[0].text = new_text
        for run in para.runs[1:]:
            run.text = ""
    else:
        para.text = new_text


def main():
    doc = Document(DOCX)
    updated = False
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("摘要："):
            replace_para_text(para, ABSTRACT)
            updated = True
        elif text.startswith("关键词："):
            replace_para_text(para, KEYWORDS)
            updated = True

    if not updated:
        raise SystemExit("Abstract paragraph not found in docx")

    doc.save(DOCX)
    print("Updated abstract in", DOCX)


if __name__ == "__main__":
    main()
