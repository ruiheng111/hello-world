#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn

SRC = "/home/ubuntu/Desktop/1997亚洲金融危机（章节正文）.docx"
OUT_DESKTOP = "/home/ubuntu/Desktop/新建 DOCX 文档.docx"
OUT_WORKSPACE = "/workspace/新建 DOCX 文档.docx"

REPLACEMENTS = [
    ("两种致命错配", "两种根本性错配"),
    (
        "并在不同程度上开放资本账户。从效率角度看",
        "并在不同程度上开放资本账户（Demirgüç-Kunt和Detragiache，1999）。从效率角度看",
    ),
    (
        "到1996年，脆弱性指标已相当清晰。泰国、印度尼西亚与菲律宾的短期外债规模接近甚至超过外汇储备；韩国虽然拥有更大的经济体量",
        "到1996年，脆弱性指标已相当清晰。泰国的短期外债规模已接近甚至超过外汇储备，印度尼西亚与菲律宾的外部脆弱性亦明显上升；韩国虽然拥有更大的经济体量",
    ),
    (
        "1997年5月，泰国财政部长公开承认政府难以继续捍卫汇率",
        "1997年5月，泰国财长公开承认政府难以继续捍卫汇率",
    ),
    ("区域传染并非神秘现象，其渠道至少包括三方面", "区域传染并不难以理解，其渠道至少包括三方面"),
    ("而非仅停留在投机攻击的表层叙事", "而非仅停留在投机攻击的表层解释"),
    (
        "为理解这一现象提供了重要理论工具：当银行体系本身成为冲击对象时",
        "为理解这一现象提供了重要参照：当银行体系本身成为冲击对象时",
    ),
    ("以向国际市场发送财政纪律与改革承诺的信号", "以向国际市场传递财政纪律与改革承诺的信号"),
    (
        "共同造就了高杠杆、低透明、弱约束的银行体系。当外部流动性收紧",
        "共同造就了高杠杆、低透明、弱约束的银行体系（Minsky，1986）。当外部流动性收紧",
    ),
    (
        "为市场重新评估汇率水平提供了基本面叙事。更重要的是",
        "为市场重新评估汇率水平提供了基本面依据。更重要的是",
    ),
    (
        "Kindleberger（1978）所描述的多米诺式恐慌，在亚洲危机中得到了清晰体现",
        "Kindleberger（1978）所描述的多米诺式恐慌，在亚洲危机中表现十分明显",
    ),
    (
        "1997年亚洲危机为这一命题提供了20世纪末最具说服力的案例之一",
        "1997年亚洲危机为这一命题提供了20世纪末最具代表性的案例之一",
    ),
    (
        "根源在于东亚经济体在1990年代前半期形成的不可持续金融结构",
        "根源在于东亚经济体在1990年代形成的不可持续金融结构",
    ),
    ("从政策设计角度看，亚洲危机表明", "由此可见，亚洲危机表明"),
    (
        "任何单纯从经济模型出发的政策设计都难以充分把握其复杂性",
        "任何单纯依靠经济模型的政策设计都难以充分把握其复杂性",
    ),
]

REFS = [
    "Bernanke, B. S. (1983). Nonmonetary effects of the financial crisis in the propagation of the Great Depression. American Economic Review, 73(3), 257–276.",
    "Corsetti, G., Pesenti, P., & Roubini, N. (1999). Paper tigers? A model of the Asian crisis. European Economic Review, 43(7), 1211–1236.",
    "Demirgüç-Kunt, A., & Detragiache, E. (1999). Financial liberalization and financial fragility. In B. Pleskovic & J. E. Stiglitz (Eds.), Annual World Bank Conference on Development Economics, 1998–1999 (pp. 303–316). World Bank.",
    "Diamond, D. W., & Dybvig, P. H. (1983). Bank runs, deposit insurance, and liquidity. Journal of Political Economy, 91(3), 401–419.",
    "Eichengreen, B. (2002). Financial Crises and What to Do About Them. Oxford University Press.",
    "Fischer, S. (1998). The Asian crisis: A view from the IMF. Journal of International Money and Finance, 18(2), 167–175.",
    "Goldstein, M. (1998). The Asian Financial Crisis: Causes, Cures, and Systemic Implications. Peterson Institute for International Economics.",
    "Kindleberger, C. P. (1978). Manias, Panics, and Crashes: A History of Financial Crises. Basic Books.",
    "Krugman, P. (1979). A model of balance-of-payments crises. Journal of Money, Credit and Banking, 11(3), 311–325.",
    "Krugman, P. (1998). What happened to Asia? MIT Department of Economics Working Paper.",
    "Lane, T., Ghosh, A. R., Hamann, J., Phillips, S., Schulze-Ghattas, M., & Tsikata, T. (1999). IMF-supported programs in Indonesia, Korea, and Thailand: A preliminary assessment. IMF Occasional Paper No. 178. International Monetary Fund.",
    "Minsky, H. P. (1986). Stabilizing an Unstable Economy. Yale University Press.",
    "Obstfeld, M. (1994). The logic of currency crises. Cahiers Économiques et Monétaires, 43, 189–213.",
    "Stiglitz, J. E. (2002). Globalization and Its Discontents. W. W. Norton.",
    "Stiglitz, J. E., & Weiss, A. (1981). Credit rationing in markets with imperfect information. American Economic Review, 71(3), 393–410.",
]


def set_run_font(run, name_cn="宋体", name_en="Times New Roman", size=10.5, bold=False):
    run.font.name = name_en
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name_cn)
    run.font.size = Pt(size)
    run.bold = bold


def replace_para_text(para, new_text):
    if para.runs:
        para.runs[0].text = new_text
        for r in para.runs[1:]:
            r.text = ""
    else:
        para.text = new_text


def apply_replacements(doc):
    for para in doc.paragraphs:
        text = para.text
        for old, new in REPLACEMENTS:
            if old in text:
                text = text.replace(old, new)
        if text != para.text:
            replace_para_text(para, text)


def update_references(doc):
    ref_indices = []
    for i, p in enumerate(doc.paragraphs):
        if i > 0 and doc.paragraphs[i - 1].text.strip() == "参考文献":
            if p.text.strip() and p.text.strip() != "图表说明":
                ref_indices.append(i)
        elif p.text.strip() == "参考文献":
            continue

    # find contiguous block after 参考文献
    start = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "参考文献":
            start = i + 1
            break
    if start is None:
        return
    ref_indices = []
    for i in range(start, len(doc.paragraphs)):
        t = doc.paragraphs[i].text.strip()
        if t == "图表说明":
            break
        if t:
            ref_indices.append(i)

    for idx, ref in zip(ref_indices, REFS):
        p = doc.paragraphs[idx]
        replace_para_text(p, ref)
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(3)
        if p.runs:
            set_run_font(p.runs[0], size=10.5)

    # clear extra old refs if any
    for idx in ref_indices[len(REFS):]:
        replace_para_text(doc.paragraphs[idx], "")


def verify(doc):
    text = "\n".join(p.text for p in doc.paragraphs)
    cites = [
        "Bernanke（1983）", "Stiglitz与Weiss（1981）", "Krugman（1979）", "Obstfeld（1994）",
        "Diamond与Dybvig（1983）", "Fischer（1998）", "Stiglitz（2002）", "Lane等（1999）",
        "Krugman（1998）", "Corsetti、Pesenti与Roubini（1999）", "Goldstein（1998）",
        "Kindleberger（1978）", "Eichengreen（2002）", "Demirgüç-Kunt和Detragiache（1999）",
        "Minsky（1986）",
    ]
    missing = [c for c in cites if c not in text]
    print("Missing in-text citations:", missing)
    ref_text = "\n".join(p.text for p in doc.paragraphs if "American Economic Review" in p.text or "Oxford University Press" in p.text or "IMF Occasional" in p.text or "Yale University Press" in p.text or "Peterson Institute" in p.text or "Basic Books" in p.text or "W. W. Norton" in p.text or "World Bank" in p.text or "MIT Department" in p.text)
    if "et al." in ref_text:
        print("WARNING: et al. still in references")


def main():
    doc = Document(SRC)
    apply_replacements(doc)
    update_references(doc)
    verify(doc)
    doc.save(OUT_DESKTOP)
    doc.save(OUT_WORKSPACE)
    print("Saved:", OUT_DESKTOP)


if __name__ == "__main__":
    main()
