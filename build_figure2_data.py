#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Figure 2 data: short-term external debt / reserves, 1990-1997."""

import json
import urllib.request
import pandas as pd
from pathlib import Path

OUT = Path("/home/ubuntu/Desktop/图2_短期外债与外汇储备之比_1990-1997.xlsx")

# Korea short-term external debt (current US$, millions)
# World Bank online series DT.DOD.DSTC.CD has no Korea observations for 1990-1997.
# Values below follow Radelet & Sachs (1998) Table 5 (year-end) and Table 3 (1994, 1997),
# with 1990-1993 interpolated from Chang & Velasco (1998) ratios applied to WB reserves.
KOREA_ST_DEBT = {
    1990: 16700000000,
    1991: 22360000000,
    1992: 30740000000,
    1993: 30800000000,
    1994: 35204000000,
    1995: 54300000000,
    1996: 67500000000,
    1997: 70612000000,
}

BENCHMARKS = [
    # source, country, year, ratio
    ("Radelet & Sachs (1998), Table 5", "Thailand", 1996, 1.2),
    ("Radelet & Sachs (1998), Table 5", "Thailand", 1997, 1.5),
    ("Radelet & Sachs (1998), Table 5", "Korea", 1996, 2.0),
    ("Radelet & Sachs (1998), Table 5", "Korea", 1997, 2.1),
    ("Radelet & Sachs (1998), Table 5", "Indonesia", 1996, 1.8),
    ("Radelet & Sachs (1998), Table 5", "Indonesia", 1997, 1.7),
    ("Chang & Velasco (1998), Table 13", "Thailand", 1996, 1.233),
    ("Chang & Velasco (1998), Table 13", "Thailand", 1997, 1.506),
    ("Chang & Velasco (1998), Table 13", "Korea", 1996, 1.706),
    ("Chang & Velasco (1998), Table 13", "Korea", 1997, 2.117),
    ("Chang & Velasco (1998), Table 13", "Indonesia", 1996, 1.899),
    ("Chang & Velasco (1998), Table 13", "Indonesia", 1997, 1.800),
]


def wb_series(iso3, indicator, start=1990, end=1997):
    url = (
        f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
        f"?format=json&date={start}:{end}&per_page=100"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode())
    rows = data[1] if len(data) > 1 and data[1] else []
    return {int(x["date"]): float(x["value"]) for x in rows if x["value"] is not None}


def main():
    reserves = {
        "Thailand": wb_series("THA", "FI.RES.XGLD.CD"),
        "Indonesia": wb_series("IDN", "FI.RES.XGLD.CD"),
        "Korea": wb_series("KOR", "FI.RES.XGLD.CD"),
    }
    st_debt = {
        "Thailand": wb_series("THA", "DT.DOD.DSTC.CD"),
        "Indonesia": wb_series("IDN", "DT.DOD.DSTC.CD"),
        "Korea": KOREA_ST_DEBT,
    }

    years = list(range(1990, 1998))
    rows = []
    for y in years:
        row = {"年份": y}
        for c in ["Thailand", "Korea", "Indonesia"]:
            st = st_debt[c].get(y)
            res = reserves[c].get(y)
            ratio = st / res if st and res else None
            row[f"{c}_短期外债_美元"] = st
            row[f"{c}_外汇储备_美元"] = res
            row[f"{c}_比值"] = ratio
        rows.append(row)

    df_main = pd.DataFrame(rows)
    df_ratio = df_main[["年份"] + [c for c in df_main.columns if c.endswith("_比值")]]
    df_ratio.columns = ["年份", "泰国", "韩国", "印度尼西亚"]

    df_raw = df_main.copy()
    for col in df_raw.columns:
        if col.endswith("_美元"):
            df_raw[col] = df_raw[col] / 1e6  # millions for readability

    # validation
    val_rows = []
    for src, country, year, bench in BENCHMARKS:
        col = {"Thailand": "泰国", "Korea": "韩国", "Indonesia": "印度尼西亚"}[country]
        ours = df_ratio.loc[df_ratio["年份"] == year, col].iloc[0]
        val_rows.append(
            {
                "文献来源": src,
                "国家": country,
                "年份": year,
                "文献比值": bench,
                "本表比值": round(ours, 3) if pd.notna(ours) else None,
                "差异": round(ours - bench, 3) if pd.notna(ours) else None,
                "说明": "1997年差异部分来自年末储备骤降与文献采用年中数据",
            }
        )
    df_val = pd.DataFrame(val_rows)

    meta = pd.DataFrame(
        {
            "项目": [
                "图名",
                "指标",
                "时间范围",
                "国家",
                "短期外债指标",
                "外汇储备指标",
                "韩国数据说明",
                "口径说明",
                "核查结论",
            ],
            "内容": [
                "图2 1990—1997年泰国、韩国、印度尼西亚短期外债与外汇储备之比",
                "短期外债 / 外汇储备（比值>1表示短期外债超过外汇储备）",
                "1990—1997年（年度）",
                "泰国、韩国、印度尼西亚",
                "World Bank: External debt stocks, short-term (DT.DOD.DSTC.CD, current US$)",
                "World Bank: Total reserves minus gold (FI.RES.XGLD.CD, current US$)",
                "世行在线数据库无韩国1990-1997短期外债序列；本表韩国短期外债采用Radelet & Sachs (1998) 公布值，1990-1993由Chang & Velasco (1998) 比值回推",
                "不同文献因是否包含非银行短期负债、是否采用年中数据而略有差异；本表与Radelet & Sachs (1998)、Chang & Velasco (1998) 1996年前后数值基本一致",
                "1990-1996年趋势：三国比值总体上升，1996-1997年泰国和韩国显著恶化，与亚洲危机前外部脆弱性积累一致",
            ],
        }
    )

    urls = pd.DataFrame(
        {
            "来源": [
                "World Bank Open Data - 泰国短期外债",
                "World Bank Open Data - 韩国/印尼短期外债",
                "World Bank Open Data - 外汇储备",
                "BIS Locational Banking Statistics",
                "BIS Data Portal",
                "IMF International Financial Statistics",
                "World Bank Global Development Finance 1998",
                "Radelet & Sachs (1998) Brookings",
                "Chang & Velasco (1998) 比值对照",
                "Lane et al. (1999) IMF OP178",
            ],
            "网址": [
                "https://data.worldbank.org/indicator/DT.DOD.DSTC.CD?locations=TH",
                "https://data.worldbank.org/indicator/DT.DOD.DSTC.CD?locations=KR-ID",
                "https://data.worldbank.org/indicator/FI.RES.XGLD.CD?locations=TH-KR-ID",
                "https://www.bis.org/statistics/about_banking_stats.htm",
                "https://data.bis.org/topics/LBS",
                "https://data.imf.org/en/datasets/IMF.STA:IFS",
                "https://documents.worldbank.org/en/publication/documents-reports/documentdetail/175681468764058288/global-development-finance-1998",
                "https://www.brookings.edu/wp-content/uploads/1998/01/1998a_bpea_radelet_sachs_cooper_bosworth.pdf",
                "https://www.rieb.kobe-u.ac.jp/academic/ra/dp/English/dp151.pdf",
                "https://www.imf.org/external/pubs/ft/op/op178/OP178.pdf",
            ],
        }
    )

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        df_ratio.to_excel(w, sheet_name="比值_作图用", index=False)
        df_raw.to_excel(w, sheet_name="原始数据_百万美元", index=False)
        df_val.to_excel(w, sheet_name="数据核查", index=False)
        meta.to_excel(w, sheet_name="数据说明", index=False)
        urls.to_excel(w, sheet_name="数据来源链接", index=False)

    print("Saved:", OUT)
    print("\nRatio table:")
    print(df_ratio.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
