#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate revised 1997 Asian Financial Crisis paper as docx."""

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

CONTENT = r'''
TITLE|1997年亚洲金融危机：成因、传导与政策启示
SUBTITLE|——基于货币危机理论与金融脆弱性框架的分析

H1|摘要
P|1997—1998年亚洲金融危机是二战后最具代表性的新兴市场经济体系统性危机之一，兼具货币危机、银行危机与资本账户危机的复合特征。本文在货币危机理论与金融脆弱性分析框架下，系统梳理危机爆发的制度背景、脆弱性积累机制、冲击传导路径及政策应对，并评估其经济后果。研究表明：在固定汇率制与金融自由化并行的制度组合下，短期外债过度积累、银行体系隐性担保所诱发的道德风险，以及本币贬值预期的自我实现，共同构成危机爆发的核心逻辑；危机经由贸易联系、竞争性贬值与银行同业及投资组合渠道迅速跨境传染。IMF主导的紧缩性救助在稳定汇率与恢复市场信心方面取得一定成效，但也因忽视银行重组与社会保障而加剧短期衰退。本文认为，维护宏观审慎约束、建立灵活的汇率安排与充足的外汇储备缓冲、以及将资本流动管理纳入政策工具箱，是防范类似危机的关键启示。
KW|亚洲金融危机；货币危机；金融脆弱性；资本流动；传染效应

H1|一、引言
P|1997年7月，泰铢率先放弃与美元的固定联系，随后印度尼西亚、韩国、马来西亚、菲律宾等经济体相继遭遇货币贬值、资本外逃与银行体系动荡，亚洲金融危机由此全面爆发。据国际货币基金组织（International Monetary Fund, IMF）估计，危机期间受冲击经济体的国内生产总值（Gross Domestic Product, GDP）平均下降约5%—10%，部分国家陷入深度衰退（Fischer, 1998）。与1970—1980年代的拉美债务危机不同，亚洲危机发生在宏观基本面看似稳健、财政赤字与通胀均处于可控水平的经济体，这一「谜题」促使学界重新审视货币危机理论的适用边界，并推动金融脆弱性、道德风险与银行中介功能研究的发展（Kaminsky & Reinhart, 1999; Corsetti, Pesenti & Roubini, 1999）。
P|本文遵循金融危机分析的经典逻辑——界定危机类型、识别制度背景与脆弱性来源、解析爆发与传导机制、评估政策应对与后果——对1997年亚洲金融危机进行系统梳理。在理论层面，本文将第一代与第二代货币危机模型、明斯基（Minsky, 1986）的金融不稳定假说，以及银行挤兑与资产负债表渠道（Bernanke & Gertler, 1989; Kiyotaki & Moore, 1997）纳入统一分析框架；在实证层面，援引危机期间的关键宏观金融指标与已有计量研究，以增强论证的可检验性与政策含义的清晰度。全文结构如下：第二节界定危机类型并描述宏观制度环境；第三节分析危机前脆弱性的形成；第四节讨论爆发、演变与跨境传染；第五节评估政策应对；第六节总结经济后果与政策启示。

H1|二、危机类型界定与宏观制度背景
H2|（一）危机类型的复合性
P|按照IMF的分类标准，金融危机可区分为货币危机、银行危机、主权债务危机与系统性危机（IMF, 1998）。1997年亚洲危机并非单一类型的货币冲击，而是典型的「孪生危机」（twin crises）：本币大幅贬值与银行体系大面积资不抵债几乎同步发生（Kaminsky & Reinhart, 1999）。以泰国为例，1997年7月至1998年1月泰铢对美元贬值超过50%；同期金融机构关闭与重组规模占GDP比重显著上升。韩国、印度尼西亚的情形更为严峻：大型财阀（chaebol）与银行交叉持股、隐性担保下的过度杠杆，使汇率冲击迅速转化为银行破产潮与信贷紧缩（Claessens, Djankov & Xu, 2000）。因此，理解亚洲危机必须同时关注外汇市场与银行中介部门的联动，而非仅套用单一维度的货币危机模型。
H2|（二）「亚洲奇迹」与制度环境
P|危机前，东亚经济体长期保持高增长、低通胀与相对平衡的财政收支，被世界银行（World Bank, 1993）概括为「东亚奇迹」。然而，高增长背后存在深刻的结构性特征：第一，普遍实行盯住美元或一篮子货币的固定（或爬行盯住）汇率制度，在资本账户逐步开放的情况下，货币政策独立性受到「不可能三角」约束（Obstfeld & Rogoff, 1995）；第二，1980年代末至1990年代中期，区域内金融自由化与资本账户开放加速，短期外债占外汇储备比重持续攀升（Reinhart & Rogoff, 2009）；第三，银行主导型金融体系中，政府对银行与大型企业的隐性担保降低了风险定价，激励了过度借贷与资产价格泡沫（Corsetti, Pesenti & Roubini, 1999）。这些制度安排在全球低利率与日本资本输出的宏观环境下，共同造就了表面稳健、实则脆弱的金融平衡。

H1|三、危机前脆弱性的形成机制
H2|（一）固定汇率、资本流动与短期外债
P|克鲁格曼（Krugman, 1979）的第一代货币危机模型强调，在固定汇率制下，财政赤字引发的国内信贷扩张将侵蚀外汇储备，最终导致盯住汇率不可持续。亚洲经济体在危机前财政状况总体尚可，但该模型的核心机制——国内信贷膨胀与储备消耗——仍以另一种形式出现：私人部门而非公共部门成为杠杆主体。1990年代初，泰国、马来西亚、印度尼西亚等国短期外债快速增长，且大量用于非贸易部门（房地产、股市）而非出口创汇部门（Radelet & Sachs, 1998）。当本币维持高估、经常账户赤字扩大时，固定汇率安排使市场形成「不可持续」的预期，为投机性攻击埋下伏笔。
H2|（二）道德风险、银行脆弱性与安全边界
P|明斯基（Minsky, 1986）的金融不稳定假说指出，在经济扩张期，融资结构从「对冲性」融资向「投机性」与「庞氏」融资演化，系统脆弱性内生积累。亚洲案例提供了该假说的典型证据：在政府隐性担保与裙带关系（crony capitalism）下，银行对关联企业发放低标准贷款，企业净值对资产价格的敏感度上升。戴蒙德与迪布维格（Diamond & Dybvig, 1983）的银行挤兑模型进一步表明，一旦公众对银行偿付能力产生怀疑，即使基本面尚可的银行也可能因流动性枯竭而倒闭，危机具有自我实现的特征。从「安全边界」（margin of safety）视角看（Minsky, 1986），危机前亚洲银行体系的资本金与拨备覆盖率已不足以吸收汇率与资产价格双重下跌的冲击；当安全边界被突破，微小的负面冲击即可触发信贷紧缩与资产抛售的恶性循环，即「金融加速器」机制（Bernanke & Gertler, 1989; Kiyotaki & Moore, 1997）。
H2|（三）第二代货币危机：预期与自我实现
P|奥布斯特菲尔德（Obstfeld, 1996）的第二代货币危机模型强调，即使不存在严重的基本面失衡，政府维护固定汇率的成本（如提高利率、收紧信贷）与放弃盯住的收益（恢复竞争力、扩张货币）之间的权衡，也可能使贬值预期自我实现。1997年5月，泰国央行对国际投机资本的防御已显著推高国内利率并挤压银行体系；当市场意识到政府无法无限消耗外汇储备时，攻击从「试探性」转为「协同性」（Obstfeld, 1996）。这一机制解释了为何危机前宏观指标「看似健康」的经济体仍可能迅速陷入货币崩溃。

H1|四、危机爆发、演变与跨境传染
H2|（一）触发事件与爆发时序
P|危机的直接导火索是1997年2—7月间对泰铢的投机攻击。1997年7月2日，泰国央行宣布放弃固定汇率制，泰铢当日贬值近20%，标志着危机正式爆发。随后，菲律宾比索、马来西亚林吉特、印度尼西亚盾、韩国韩元在1997年下半年至1998年初相继大幅贬值。印度尼西亚盾跌幅最为剧烈，1997年7月至1998年1月累计贬值超过70%，并伴随严重的社会政治动荡（Claessens, Djankov & Xu, 2000）。香港特区政府在1997年8月经历「恒指保卫战」，虽成功捍卫联系汇率制，但区域信心仍持续恶化。
H2|（二）传染渠道
P|Kaminsky和Reinhart（1999）将金融危机传染区分为基本面关联、竞争性贬值与纯粹恐慌三类。亚洲危机中三种渠道均发挥作用：第一，贸易联系与共同冲击——区域内出口结构相似，日元贬值与全球电子产品需求下滑同时冲击各国经常账户；第二，竞争性贬值——一国贬值迫使贸易伙伴跟进，形成「以邻为壑」的连锁反应（Glick & Rose, 1999）；第三，投资组合与银行同业渠道——国际投资者在对一国失去信心后，基于区域风险重新定价而同步撤资，使基本面较弱但原本独立的经济体（如韩国）亦遭波及（Baig & Goldfajn, 1999）。Calvo和Reinhart（2000）进一步指出，在信息不完全环境下，「传染」往往体现为共同基金规则与风险限额所驱动的「非基本面」资本流动，这加剧了危机的广度与深度。

H1|五、政策应对与国际救助
H2|（一）各国国内政策
P|危机爆发后，各国政策反应呈现显著差异。马来西亚在1998年实施资本管制并固定林吉特汇率，短期内稳定了金融市场，但也引发关于政策有效性与道德风险的争议（Kaplan & Rodrik, 2002）。韩国在IMF援助条件下推进金融与财阀改革，关闭问题银行、注入公共资本并加强信息披露，为后续复苏奠定基础。印度尼西亚因政治不稳定与银行重组滞后，调整过程更为痛苦。总体而言，成功的应对需同时包含：汇率安排的及时调整、问题金融机构的迅速重组、以及财政资源对存款人保护与社会稳定的支持（Fischer, 1998）。
H2|（二）IMF救助方案及其争议
P|IMF向泰国、印度尼西亚、韩国等提供了大规模备用贷款，条件包括财政紧缩、利率维持高位以稳定汇率、关闭问题金融机构及推进结构改革（IMF, 1998）。Fischer（1998）认为，在信心崩溃的极端情形下，紧缩性政策对于恢复市场信任与阻止资本外逃具有必要性。然而，Stiglitz（2002）等批评者指出，IMF「一刀切」的紧缩方案未充分区分流动性危机与偿付能力危机，高利率政策加剧了企业破产与银行不良贷款上升，短期社会成本过高。后续研究倾向于认为，IMF方案在稳定汇率方面有效，但在银行重组时机选择与社会保障安排方面存在明显不足（Lane et al., 1999; Corsetti, Pesenti & Roubini, 1999）。

H1|六、经济后果与政策启示
H2|（一）经济后果
P|危机造成严重的短期经济收缩与社会代价。1998年，印度尼西亚、泰国、马来西亚的GDP分别下降13.1%、7.6%和7.4%（World Bank, 1999）。失业率上升、贫困率反弹，部分国家的政治稳定亦受到冲击。从中长期看，韩国、泰国等国通过银行重组、公司治理改革与汇率制度调整，于1999年后逐步恢复增长；但印度尼西亚的复苏更为缓慢，说明制度能力与政治约束在危机后调整中的关键作用（Claessens, Djankov & Xu, 2000）。
H2|（二）政策启示
P|第一，宏观审慎管理应置于与货币政策、财政政策同等重要的位置。危机表明，在低通胀与财政平衡条件下，私人部门杠杆与外币负债仍可能构成系统性风险（Borio, 2003）。第二，汇率制度需与资本账户开放程度相匹配；完全固定汇率在面临大规模资本流动时维护成本过高，更具弹性的安排有助于吸收冲击（Obstfeld & Rogoff, 1995）。第三，应建立充足的外汇储备与区域金融合作机制——2000年成立的清迈倡议（Chiang Mai Initiative）及此后东盟与中日韩（ASEAN+3）多边化安排，正是对亚洲危机教训的直接回应（Henning, 2009）。第四，银行监管须覆盖关联贷款、外币敞口与期限错配，消除隐性担保所诱发的道德风险（Tornell & Westermann, 2005）。第五，国际救助应区分流动性支持与偿付能力调整，避免在银行体系未完成重组前过度紧缩实体部门（Sachs, 1998）。

H1|七、结论
P|1997年亚洲金融危机是固定汇率、金融自由化与银行体系道德风险相互叠加的产物，其爆发与传导机制无法由单一货币危机模型完整解释，而需在孪生危机与金融加速器框架下加以理解。危机对东亚乃至全球金融治理产生了深远影响：它推动了宏观审慎政策、外汇储备积累、区域金融合作以及国际货币基金组织改革议程的讨论。对于当前新兴市场经济体而言，在全球利率上行、资本流动波动加剧的环境下，重新检视这一案例仍具有重要的现实意义：宏观指标的表面稳健并不等同于金融体系的实质稳健，而预期管理、银行中介治理与跨境传染防控，始终是金融危机防范不可省略的环节。
'''

REFS = [
    "Baig, T., & Goldfajn, I. (1999). Financial market contagion in the Asian crisis. IMF Staff Papers, 46(2), 167–195.",
    "Bernanke, B. S., & Gertler, M. (1989). Agency costs, net worth, and business fluctuations. American Economic Review, 79(1), 14–31.",
    "Borio, C. (2003). Towards a macroprudential framework for financial supervision and regulation? BIS Working Papers, No. 128.",
    "Calvo, G. A., & Reinhart, C. M. (2000). When capital inflows come to a sudden stop: Consequences and policy options. In P. Kenen & A. Swoboda (Eds.), Reforming the International Monetary and Financial System (pp. 175–201). IMF.",
    "Claessens, S., Djankov, S., & Xu, L. C. (2000). Corporate performance in the East Asian financial crisis. World Bank Research Observer, 15(1), 23–46.",
    "Corsetti, G., Pesenti, P., & Roubini, N. (1999). What caused the Asian currency and financial crisis? Japan and the World Economy, 11(3), 305–373.",
    "Diamond, D. W., & Dybvig, P. H. (1983). Bank runs, deposit insurance, and liquidity. Journal of Political Economy, 91(3), 401–419.",
    "Fischer, S. (1998). The IMF and the Asian crisis. Lecture at UCLA, March 20, 1998.",
    "Glick, R., & Rose, A. K. (1999). Contagion and trade: Why are currency crises regional? Journal of International Money and Finance, 18(4), 603–617.",
    "Henning, C. R. (2009). The future of the Chiang Mai Initiative: An Asian monetary fund? Peterson Institute for International Economics Policy Brief, No. 09-5.",
    "International Monetary Fund. (1998). World Economic Outlook, May 1998: Financial crises—Causes and indicators. IMF.",
    "Kaplan, E., & Rodrik, D. (2002). Did the Malaysian capital controls work? In S. Edwards & J. A. Frankel (Eds.), Preventing Currency Crises in Emerging Markets (pp. 393–440). University of Chicago Press.",
    "Kaminsky, G. L., & Reinhart, C. M. (1999). The twin crises: The causes of banking and balance-of-payments problems. American Economic Review, 89(3), 473–500.",
    "Kiyotaki, N., & Moore, J. (1997). Credit cycles. Journal of Political Economy, 105(2), 211–248.",
    "Krugman, P. (1979). A model of balance-of-payments crises. Journal of Money, Credit and Banking, 11(3), 311–325.",
    "Lane, T., et al. (1999). IMF-supported programs in Indonesia, Korea and Thailand: A preliminary assessment. IMF Occasional Paper, No. 178.",
    "Minsky, H. P. (1986). Stabilizing an Unstable Economy. Yale University Press.",
    "Obstfeld, M. (1996). Models of currency crises with self-fulfilling features. European Economic Review, 40(3–5), 1037–1047.",
    "Obstfeld, M., & Rogoff, K. (1995). Exchange rate dynamics redux. Journal of Political Economy, 103(3), 624–660.",
    "Radelet, S., & Sachs, J. D. (1998). The East Asian financial crisis: Diagnosis, remedies, prospects. Brookings Papers on Economic Activity, 1998(1), 1–90.",
    "Reinhart, C. M., & Rogoff, K. S. (2009). This Time Is Different: Eight Centuries of Financial Folly. Princeton University Press.",
    "Sachs, J. D. (1998). The IMF and the Asian flu. American Prospect, 37, 17–23.",
    "Stiglitz, J. E. (2002). Globalization and Its Discontents. W. W. Norton.",
    "Tornell, A., & Westermann, F. (2005). Boom–bust cycles and the balance sheet effect. Journal of International Economics, 66(1), 1–20.",
    "World Bank. (1993). The East Asian Miracle: Economic Growth and Public Policy. Oxford University Press.",
    "World Bank. (1999). East Asia: The Road to Recovery. World Bank.",
]


def set_run_font(run, name_cn="宋体", name_en="Times New Roman", size=12, bold=False):
    run.font.name = name_en
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name_cn)
    run.font.size = Pt(size)
    run.bold = bold


def add_para(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=0.74, size=12, bold=False):
    p = doc.add_paragraph()
    p.alignment = align
    if first_line_indent:
        p.paragraph_format.first_line_indent = Cm(first_line_indent)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def add_heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    h.clear()
    run = h.add_run(text)
    sizes = {1: 16, 2: 14}
    set_run_font(run, size=sizes.get(level, 12), bold=True)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
    return h


def build_document():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)

    for line in CONTENT.strip().splitlines():
        if not line.strip():
            continue
        kind, text = line.split("|", 1)
        if kind == "TITLE":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            set_run_font(run, name_cn="黑体", size=18, bold=True)
            p.paragraph_format.space_after = Pt(12)
        elif kind == "SUBTITLE":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            set_run_font(run, size=14)
            p.paragraph_format.space_after = Pt(18)
        elif kind == "H1":
            add_heading(doc, text, level=1)
        elif kind == "H2":
            add_heading(doc, text, level=2)
        elif kind == "P":
            add_para(doc, text)
        elif kind == "KW":
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Cm(0.74)
            p.paragraph_format.line_spacing = 1.5
            r1 = p.add_run("关键词：")
            set_run_font(r1, bold=True)
            r2 = p.add_run(text)
            set_run_font(r2)

    add_heading(doc, "参考文献", level=1)
    for ref in REFS:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(ref)
        set_run_font(run, size=10.5)

    return doc


if __name__ == "__main__":
    out = "/home/ubuntu/Desktop/1997亚洲金融危机（修订稿）.docx"
    build_document().save(out)
    print(f"Saved to {out}")
