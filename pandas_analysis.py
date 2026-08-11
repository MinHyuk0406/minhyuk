"""Create presentation-ready EDA outputs with Pandas.

This is a reproducible analysis step for the three Seoul commercial-area CSVs.
It does not run in Vercel; the deployed dashboard continues to read the small
static JSON package made by build_dashboard.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_dashboard import find_sources


KEYS = ["quarter", "dong_code", "industry_code"]


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric values and preserve missing values."""
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_source_frames(input_dir: Path) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Read the original CSVs by their stable positional schema."""
    sources = find_sources(input_dir)
    raw_stores = pd.read_csv(sources["stores"], encoding="cp949", dtype=str)
    raw_sales = pd.read_csv(sources["sales"], encoding="cp949", dtype=str)
    raw_population = pd.read_csv(sources["population"], encoding="cp949", dtype=str)

    stores = raw_stores.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]].copy()
    stores.columns = [
        "quarter", "dong_code", "dong", "industry_code", "industry", "store_count",
        "franchise_count", "open_rate", "open_count", "close_rate", "close_count",
    ]
    stores = numeric(stores, [
        "store_count", "franchise_count", "open_rate", "open_count", "close_rate", "close_count",
    ])

    sales = raw_sales.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
    sales.columns = ["quarter", "dong_code", "dong", "industry_code", "industry", "sales_won", "transactions"]
    sales = numeric(sales, ["sales_won", "transactions"])

    population = raw_population.iloc[:, [0, 1, 3]].copy()
    population.columns = ["quarter", "dong_code", "floating_population"]
    population = numeric(population, ["floating_population"])
    # The dashboard uses one population observation per quarter and dong.
    population = population.drop_duplicates(["quarter", "dong_code"], keep="last")

    return sources, {"stores": stores, "sales": sales, "population": population}


def quality_rows(frames: dict[str, pd.DataFrame], merged: pd.DataFrame) -> pd.DataFrame:
    """Return auditable source, key-duplicate, and missing-value summaries."""
    rows = []
    for name, frame in frames.items():
        applicable_keys = [key for key in KEYS if key in frame.columns]
        rows.append({
            "dataset": name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "duplicate_key_rows": int(frame.duplicated(applicable_keys, keep=False).sum()),
            "missing_cells": int(frame.isna().sum().sum()),
        })
    rows.append({
        "dataset": "merged_analysis_base",
        "rows": len(merged),
        "columns": len(merged.columns),
        "duplicate_key_rows": int(merged.duplicated(KEYS, keep=False).sum()),
        "missing_cells": int(merged.isna().sum().sum()),
    })
    return pd.DataFrame(rows)


def build_merged(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join store, sales, and population data and add presentation variables."""
    stores, sales, population = frames["stores"], frames["sales"], frames["population"]
    merged = stores.merge(
        sales.drop(columns=["dong", "industry"]), on=KEYS, how="inner", validate="one_to_one"
    ).merge(population, on=["quarter", "dong_code"], how="inner", validate="many_to_one")
    merged = merged.loc[merged["store_count"].gt(0)].copy()
    merged["sales_per_store_won"] = merged["sales_won"] / merged["store_count"]
    merged["net_entry_rate"] = merged["open_rate"] - merged["close_rate"]
    merged["population_per_store"] = merged["floating_population"] / merged["store_count"]
    return merged


def dashboard_quadrant_summary(dashboard_path: Path) -> pd.DataFrame:
    """Summarize the current dashboard's ML-derived quadrant labels with Pandas."""
    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    records = pd.DataFrame(payload["records"])
    summary = (
        records.groupby(["q", "ql"], dropna=False)
        .agg(
            행정동업종_조합수=("dc", "size"),
            평균_현재_폐업률=("cr", "mean"),
            평균_개업률=("or", "mean"),
            평균_순점포증감률=("ne", "mean"),
            평균_점포당매출_원=("ss", "mean"),
            평균_AI다음분기폐업률=("mp", "mean"),
        )
        .reset_index()
        .sort_values("행정동업종_조합수", ascending=False)
    )
    return summary


def write_report(output_dir: Path, sources: dict[str, Path], quality: pd.DataFrame,
                 quarterly: pd.DataFrame, quadrants: pd.DataFrame) -> None:
    """Write concise Korean notes that can be used directly in a presentation."""
    merged_row = quality.loc[quality["dataset"].eq("merged_analysis_base")].iloc[0]
    lines = [
        "# Pandas 기반 데이터 분석 요약",
        "",
        "## 분석 과정",
        "",
        "1. Pandas로 점포·추정매출·길단위 유동인구 CSV를 CP949 인코딩으로 불러왔습니다.",
        "2. 분기·행정동·서비스업종을 기준으로 점포와 매출을 결합하고, 분기·행정동 기준으로 유동인구를 결합했습니다.",
        "3. 점포당 매출, 순점포 증감률(개업률−폐업률), 점포당 유동인구를 파생변수로 생성했습니다.",
        "4. 현재 대시보드의 AI 예측 결과를 Pandas groupby로 집계해 4분면별 특징을 비교했습니다.",
        "",
        "## 데이터 품질",
        "",
        f"- 결합 후 분석 가능 행: {int(merged_row['rows']):,}건",
        f"- 결합 후 키 중복 행: {int(merged_row['duplicate_key_rows']):,}건",
        f"- 분석 기준 분기 수: {quarterly['quarter'].nunique():,}개",
        "",
        "## 4분면 요약",
        "",
        "| 구분 | 행정동·업종 조합 수 | 평균 현재 폐업률 | 평균 개업률 | 평균 순점포 증감률 | 평균 AI 다음 분기 폐업률 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in quadrants.iterrows():
        lines.append(
            f"| {row['ql']} | {int(row['행정동업종_조합수']):,} | {row['평균_현재_폐업률']:.2f}% | "
            f"{row['평균_개업률']:.2f}% | {row['평균_순점포증감률']:.2f}%p | {row['평균_AI다음분기폐업률']:.2f}% |"
        )
    lines.extend([
        "",
        "## 원천 파일",
        "",
        *(f"- {name}: `{path.name}`" for name, path in sources.items()),
        "",
        "주의: 본 분석은 행정동·업종 단위의 공개 통계 비교이며 개별 점포의 폐업 사유나 성공을 판정하지 않습니다.",
    ])
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pandas EDA for OpenSafe AI.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing the 3 Seoul CSVs")
    parser.add_argument("--dashboard-data", type=Path, default=Path("data/dashboard-data.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    sources, frames = load_source_frames(args.input_dir)
    merged = build_merged(frames)
    quality = quality_rows(frames, merged)
    quarterly = (
        merged.groupby("quarter", as_index=False)
        .agg(
            행정동업종_조합수=("dong_code", "size"),
            평균_폐업률=("close_rate", "mean"),
            평균_개업률=("open_rate", "mean"),
            평균_순점포증감률=("net_entry_rate", "mean"),
            평균_점포당매출_원=("sales_per_store_won", "mean"),
        )
        .sort_values("quarter")
    )
    quadrants = dashboard_quadrant_summary(args.dashboard_data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.output_dir / "data_quality_summary.csv", index=False, encoding="utf-8-sig")
    quarterly.to_csv(args.output_dir / "quarterly_market_summary.csv", index=False, encoding="utf-8-sig")
    quadrants.to_csv(args.output_dir / "quadrant_summary.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, sources, quality, quarterly, quadrants)
    print(f"Pandas analysis complete: {len(merged):,} merged rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
