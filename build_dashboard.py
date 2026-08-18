"""Build a small, static data package for the OpenSafe AI dashboard.

The source CSVs are intentionally identified by schema rather than filename so
the script works with the original Korean filenames without editing paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


SCHEMAS = {25: "population", 53: "sales", 12: "stores"}


def number(value: str) -> int:
    """Parse the integer values used in the Seoul commercial-area CSVs."""
    value = (value or "0").strip().replace(",", "")
    try:
        return int(float(value))
    except ValueError:
        return 0


def find_sources(input_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in input_dir.glob("*.csv"):
        try:
            with path.open("r", encoding="cp949", newline="") as source:
                columns = len(next(csv.reader(source)))
        except UnicodeDecodeError:
            # Downloads may contain unrelated UTF-8 CSVs. They are not inputs.
            continue
        kind = SCHEMAS.get(columns)
        if kind:
            if kind in found:
                raise RuntimeError(f"More than one {kind} CSV was found in {input_dir}.")
            found[kind] = path

    missing = set(SCHEMAS.values()) - set(found)
    if missing:
        raise RuntimeError(
            "Could not identify all three input files. Missing schemas: " + ", ".join(sorted(missing))
        )
    return found


def read_population(path: Path) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="cp949", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                result[(row[0], row[1])] = number(row[3])
    return result


def read_stores(path: Path) -> dict[tuple[str, str, str], dict]:
    result: dict[tuple[str, str, str], dict] = {}
    with path.open("r", encoding="cp949", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        for row in reader:
            if len(row) < 12:
                continue
            key = (row[0], row[1], row[3])
            result[key] = {
                "dong": row[2],
                "industry": row[4],
                "stores": number(row[5]),
                "franchise": number(row[7]),
                "open_rate": number(row[8]),
                "opens": number(row[9]),
                "close_rate": number(row[10]),
                "closes": number(row[11]),
            }
    return result


def read_sales(path: Path) -> dict[tuple[str, str, str], dict]:
    result: dict[tuple[str, str, str], dict] = {}
    with path.open("r", encoding="cp949", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            key = (row[0], row[1], row[3])
            result[key] = {
                "dong": row[2],
                "industry": row[4],
                "sales": number(row[5]),
                "transactions": number(row[6]),
            }
    return result


def percentile_ranks(values: list[float]) -> list[float]:
    """Return a 0..1 empirical percentile for every input, with tied ranks."""
    if len(values) <= 1:
        return [0.5] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.5] * len(values)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[start][1]:
            end += 1
        rank = (start + end) / 2 / (len(ordered) - 1)
        for position in range(start, end + 1):
            result[ordered[position][0]] = rank
        start = end + 1
    return result


def risk_label(score: int) -> str:
    if score >= 75:
        return "높음"
    if score >= 60:
        return "주의"
    if score >= 40:
        return "관심"
    return "안정"


NUMERIC_FEATURES = [
    "current_close_rate",
    "current_open_rate",
    "log_store_count",
    "log_store_density",
    "log_sales_per_store",
    "sales_per_store_growth",
    "franchise_share",
    "log_transactions_per_store",
    "growth_available",
]

FEATURE_LABELS = [
    "최근 폐업률",
    "최근 개업률",
    "점포 수",
    "점포 밀도",
    "점포당 매출",
    "점포당 매출 증감",
    "프랜차이즈 비중",
    "점포당 거래건수",
    "매출 추세 보유 여부",
]


def model_feature_values(record: dict) -> list[float]:
    """Numerical features observable at the start of a prediction horizon."""
    stores = max(record["s"], 1)
    growth = record["g"] if record["g"] is not None else 0.0
    return [
        float(record["cr"]),
        float(record["or"]),
        math.log1p(stores),
        math.log1p(stores * 1_000_000 / max(record["p"], 1)),
        math.log1p(max(record["ss"], 0)),
        max(-100.0, min(100.0, float(growth))),
        record["f"] / stores,
        math.log1p(max(record["tx"] / stores, 0)),
        1.0 if record["g"] is not None else 0.0,
    ]


def fit_ridge(rows: list[dict], industry_codes: list[str], alpha: float) -> dict:
    """Fit a regularized linear ML model using only NumPy.

    An industry one-hot term learns different category baselines, while the
    numeric inputs quantify the location's current business conditions.
    """
    category_index = {code: index for index, code in enumerate(industry_codes)}
    numeric = np.asarray([model_feature_values(row) for row in rows], dtype=float)
    means = numeric.mean(axis=0)
    scales = numeric.std(axis=0)
    scales[scales < 1e-9] = 1.0
    standardized = (numeric - means) / scales
    category = np.zeros((len(rows), len(industry_codes)), dtype=float)
    for index, row in enumerate(rows):
        category[index, category_index[row["ic"]]] = 1.0
    design = np.column_stack([np.ones(len(rows)), standardized, category])
    target = np.asarray([row["target"] for row in rows], dtype=float)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0  # Never penalize the intercept.
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {
        "weights": weights,
        "means": means,
        "scales": scales,
        "category_index": category_index,
        "alpha": alpha,
    }


def ridge_predict(model: dict, rows: list[dict], with_contributions: bool = False):
    numeric = np.asarray([model_feature_values(row) for row in rows], dtype=float)
    standardized = (numeric - model["means"]) / model["scales"]
    category = np.zeros((len(rows), len(model["category_index"])), dtype=float)
    for index, row in enumerate(rows):
        category_index = model["category_index"].get(row["ic"])
        if category_index is not None:
            category[index, category_index] = 1.0
    design = np.column_stack([np.ones(len(rows)), standardized, category])
    predictions = design @ model["weights"]
    if not with_contributions:
        return predictions
    feature_weights = model["weights"][1 : 1 + len(NUMERIC_FEATURES)]
    contributions = standardized * feature_weights
    return predictions, contributions


def mean_absolute_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def root_mean_squared_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def risk_ranking_validation(rows: list[dict], predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Check whether the model's high-risk ranks also had higher future closure.

    Ranks are calculated within an industry, matching the dashboard's comparison
    unit.  The held-out target quarter is never used to fit the model.
    """
    by_industry: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_industry[row["ic"]].append(index)

    high, low = [], []
    for indexes in by_industry.values():
        ranks = percentile_ranks([float(predicted[index]) for index in indexes])
        for position, index in enumerate(indexes):
            if ranks[position] >= 0.75:
                high.append(float(actual[index]))
            if ranks[position] <= 0.25:
                low.append(float(actual[index]))

    return {
        "unit": "same_industry_quartile",
        "high_risk_count": len(high),
        "low_risk_count": len(low),
        "high_risk_actual_close_rate": round(float(np.mean(high)), 2) if high else None,
        "low_risk_actual_close_rate": round(float(np.mean(low)), 2) if low else None,
        "gap_percentage_points": round(float(np.mean(high) - np.mean(low)), 2) if high and low else None,
    }


def next_quarter_code(quarter: str) -> str:
    year, quarter_number = int(quarter[:4]), int(quarter[4])
    return f"{year}{quarter_number + 1}" if quarter_number < 4 else f"{year + 1}1"


def train_ai_model(pairs_by_quarter: dict[str, list[dict]], source_quarters: list[str], current_rows: list[dict]) -> tuple[dict, dict]:
    """Tune, evaluate, then fit the next-quarter closure-rate predictor.

    The final source quarter is held out for a genuine future-period test. It
    is only added back when fitting the version used for the current forecast.
    """
    if len(source_quarters) < 4:
        raise RuntimeError("At least five common quarters are required for ML validation.")
    industries = sorted({row["ic"] for rows in pairs_by_quarter.values() for row in rows})
    tune_train_quarters = source_quarters[:-2]
    tune_validation_quarter = source_quarters[-2]
    test_quarter = source_quarters[-1]
    tune_train = [row for quarter in tune_train_quarters for row in pairs_by_quarter[quarter]]
    tune_validation = pairs_by_quarter[tune_validation_quarter]

    candidates = [1.0, 10.0, 30.0, 100.0, 300.0]
    candidate_scores = []
    for alpha in candidates:
        candidate_model = fit_ridge(tune_train, industries, alpha)
        validation_predictions = ridge_predict(candidate_model, tune_validation)
        validation_actual = np.asarray([row["target"] for row in tune_validation], dtype=float)
        candidate_scores.append((mean_absolute_error(validation_predictions, validation_actual), alpha))
    selected_alpha = min(candidate_scores)[1]

    test_train = [row for quarter in source_quarters[:-1] for row in pairs_by_quarter[quarter]]
    test_rows = pairs_by_quarter[test_quarter]
    evaluation_model = fit_ridge(test_train, industries, selected_alpha)
    test_predictions = ridge_predict(evaluation_model, test_rows)
    test_actual = np.asarray([row["target"] for row in test_rows], dtype=float)
    baseline_predictions = np.asarray([row["cr"] for row in test_rows], dtype=float)
    risk_validation = risk_ranking_validation(test_rows, test_predictions, test_actual)

    final_train = [row for quarter in source_quarters for row in pairs_by_quarter[quarter]]
    final_model = fit_ridge(final_train, industries, selected_alpha)
    current_predictions, contributions = ridge_predict(final_model, current_rows, with_contributions=True)
    holdout_mae = mean_absolute_error(test_predictions, test_actual)

    for index, row in enumerate(current_rows):
        predicted = float(max(0.0, min(100.0, current_predictions[index])))
        row["mp"] = round(predicted, 1)
        row["mlow"] = round(max(0.0, predicted - holdout_mae), 1)
        row["mhigh"] = round(min(100.0, predicted + holdout_mae), 1)
        strongest = sorted(range(len(NUMERIC_FEATURES)), key=lambda item: abs(contributions[index, item]), reverse=True)[:3]
        row["mi"] = [[feature, "up" if contributions[index, feature] >= 0 else "down"] for feature in strongest]

    metrics = {
        "model_name": "Ridge regression",
        "prediction_target": "next_quarter_close_rate",
        "training_pairs": len(final_train),
        "selected_alpha": selected_alpha,
        "holdout_feature_quarter": test_quarter,
        "holdout_target_quarter": None,
        "holdout_mae": round(holdout_mae, 2),
        "holdout_rmse": round(root_mean_squared_error(test_predictions, test_actual), 2),
        "baseline_mae": round(mean_absolute_error(baseline_predictions, test_actual), 2),
        "tuning_mae": round(min(candidate_scores)[0], 2),
        "feature_labels": FEATURE_LABELS,
        "risk_index": {
            "method": "same_industry_rank_of_ml_predicted_next_quarter_close_rate",
            "validation": risk_validation,
        },
    }
    return metrics, final_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenSafe AI dashboard data.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing the 3 Seoul CSVs")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    args = parser.parse_args()

    sources = find_sources(args.input_dir)
    population = read_population(sources["population"])
    stores = read_stores(sources["stores"])
    sales = read_sales(sources["sales"])

    quarters = sorted({key[0] for key in stores} & {key[0] for key in sales} & {key[0] for key in population})
    if not quarters:
        raise RuntimeError("The CSVs have no common quarter.")
    latest = quarters[-1]

    # Build a sales-per-store history for matched records. It is used only for
    # trend explanation, not as a claim about a specific future closure.
    history: dict[tuple[str, str], list[list[object]]] = defaultdict(list)
    for quarter in quarters:
        for (q, dong_code, industry_code), sale in sales.items():
            if q != quarter:
                continue
            store = stores.get((q, dong_code, industry_code))
            if not store or store["stores"] <= 0:
                continue
            history[(dong_code, industry_code)].append([quarter, round(sale["sales"] / store["stores"])])

    # Prepare an observation for every common quarter. These inputs are all
    # known at that quarter and can therefore safely predict the following one.
    quarter_index = {quarter: index for index, quarter in enumerate(quarters)}
    sales_per_store: dict[tuple[str, str, str], float] = {}
    for key, sale in sales.items():
        store = stores.get(key)
        if store and store["stores"] > 0:
            sales_per_store[key] = sale["sales"] / store["stores"]

    observations_by_quarter: dict[str, list[dict]] = defaultdict(list)
    for (quarter, dong_code, industry_code), sale in sales.items():
        if quarter not in quarter_index:
            continue
        store = stores.get((quarter, dong_code, industry_code))
        flow = population.get((quarter, dong_code))
        if not store or not flow or store["stores"] <= 0:
            continue
        series = history.get((dong_code, industry_code), [])
        previous = None
        if quarter_index[quarter] > 0:
            previous_quarter = quarters[quarter_index[quarter] - 1]
            previous = sales_per_store.get((previous_quarter, dong_code, industry_code))
        growth = None
        if previous and previous > 0:
            growth = round((sale["sales"] / store["stores"] / previous - 1) * 100, 1)
        observations_by_quarter[quarter].append(
            {
                "d": store["dong"],
                "dc": dong_code,
                "i": store["industry"],
                "ic": industry_code,
                "p": flow,
                "s": store["stores"],
                "f": store["franchise"],
                "o": store["opens"],
                "or": store["open_rate"],
                "c": store["closes"],
                "cr": store["close_rate"],
                "sa": sale["sales"],
                "tx": sale["transactions"],
                "ss": round(sale["sales"] / store["stores"]),
                "ps": round(flow / store["stores"]),
                "g": growth,
                "h": series,
            }
        )

    records = observations_by_quarter[latest]
    source_quarters = quarters[:-1]
    pairs_by_quarter: dict[str, list[dict]] = defaultdict(list)
    for index, quarter in enumerate(source_quarters):
        future_quarter = quarters[index + 1]
        for record in observations_by_quarter[quarter]:
            future_store = stores.get((future_quarter, record["dc"], record["ic"]))
            if future_store:
                pair = dict(record)
                pair["target"] = future_store["close_rate"]
                pairs_by_quarter[quarter].append(pair)
    ai_metrics, _ = train_ai_model(pairs_by_quarter, source_quarters, records)
    ai_metrics["holdout_target_quarter"] = latest
    ai_metrics["forecast_quarter"] = next_quarter_code(latest)

    by_industry: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_industry[record["ic"]].append(record)

    for industry_records in by_industry.values():
        close_p = percentile_ranks([record["cr"] for record in industry_records])
        density_p = percentile_ranks([record["s"] * 1_000_000 / max(record["p"], 1) for record in industry_records])
        sales_p = percentile_ranks([record["ss"] for record in industry_records])
        growth_p = percentile_ranks([record["g"] if record["g"] is not None else 0 for record in industry_records])
        transaction_p = percentile_ranks([record["tx"] / max(record["s"], 1) for record in industry_records])
        demand_p = percentile_ranks([record["ps"] for record in industry_records])
        franchise_p = percentile_ranks([record["f"] / max(record["s"], 1) for record in industry_records])
        prediction_p = percentile_ranks([record["mp"] for record in industry_records])
        net_entry = [record["or"] - record["cr"] for record in industry_records]
        net_entry_p = percentile_ranks(net_entry)
        for index, record in enumerate(industry_records):
            # The risk index is no longer a manually weighted formula. It is the
            # same-industry rank of the Ridge model's next-quarter closure-rate
            # estimate, whose coefficients were fitted on historical quarters.
            risk = round(100 * prediction_p[index])
            record["r"] = risk
            record["rl"] = risk_label(risk)
            record["x"] = [
                round(100 * close_p[index]),
                round(100 * density_p[index]),
                round(100 * (1 - sales_p[index])),
                round(100 * (1 - growth_p[index])),
            ]
            # Keep one shared rank for both labels shown in the dashboard. This
            # prevents the relative-risk card and forecast explanation drifting
            # apart through duplicated calculations or rounding.
            record["mr"] = risk
            record["ml"] = risk_label(record["mr"])

            # Startup fit is deliberately broader than next-quarter closure
            # risk.  The same-industry percentile components turn different
            # units into comparable signals without claiming causality.
            components = {
                "stability": 100 * (1 - prediction_p[index]),
                "revenue_capacity": 100 * (
                    0.50 * sales_p[index] + 0.30 * transaction_p[index] + 0.20 * growth_p[index]
                ),
                "competition_balance": 100 * (
                    0.70 * (1 - density_p[index]) + 0.30 * (1 - franchise_p[index])
                ),
                "market_momentum": 100 * (0.60 * net_entry_p[index] + 0.40 * growth_p[index]),
                "demand_capacity": 100 * demand_p[index],
            }
            fit = (
                0.30 * components["stability"]
                + 0.25 * components["revenue_capacity"]
                + 0.20 * components["competition_balance"]
                + 0.15 * components["market_momentum"]
                + 0.10 * components["demand_capacity"]
            )
            record["fc"] = {key: round(value) for key, value in components.items()}
            record["fs"] = round(fit)
            record["fl"] = "높음" if fit >= 70 else "보통" if fit >= 45 else "낮음"

        # The opportunity axis uses net entry rather than openings alone: a
        # place where openings are high but closures are even higher is not
        # automatically a growth market.  Both axes remain same-industry
        # comparisons so categories are not distorted by industry scale.
        entry_median = statistics.median(net_entry)
        sales_median = statistics.median(record["ss"] for record in industry_records)
        for index, record in enumerate(industry_records):
            risk_high = record["mr"] >= 50
            entry_high = net_entry[index] >= entry_median
            record["ne"] = round(net_entry[index], 1)
            record["eh"] = entry_high
            if not risk_high and entry_high:
                record["q"] = "stable_growth"
                record["ql"] = "안정적 성장"
                record["qd"] = "예측 폐업 위험은 낮고 순점포 증감은 활발한 확장 후보입니다."
            elif risk_high and entry_high:
                record["q"] = "red_ocean"
                record["ql"] = "경쟁 치열"
                record["qd"] = "진입은 활발하지만 예측 폐업 위험도 높아 비용·차별화 검증이 필요합니다."
            elif not risk_high:
                record["q"] = "mature_stable"
                record["ql"] = "성숙·안정"
                record["qd"] = "예측 폐업 위험은 낮지만 순점포 증감은 낮아 현장 수요 확인이 중요합니다."
            elif record["ss"] >= sales_median and record["g"] is not None and record["g"] >= 0:
                record["q"] = "niche_candidate"
                record["ql"] = "틈새 검토"
                record["qd"] = "수축 신호 안에서도 점포당 매출과 최근 추세가 버티는지 추가 검토할 후보입니다."
            else:
                record["q"] = "contraction_risk"
                record["ql"] = "수축 위험"
                record["qd"] = "진입과 안정성 모두 약한 편으로 계약 전 수요·비용을 보수적으로 확인해야 합니다."

    industries = sorted(
        ({"code": code, "name": rows[0]["i"], "count": len(rows)} for code, rows in by_industry.items()),
        key=lambda item: item["name"],
    )
    dongs = sorted({record["d"] for record in records})
    output = {
        "meta": {
            "latest_quarter": latest,
            "available_quarters": quarters,
            "record_count": len(records),
            "dong_count": len(dongs),
            "industry_count": len(industries),
            "method": "same-industry empirical ranking",
            "ai_model": ai_metrics,
            "startup_fit": {
                "method": "same_industry_weighted_startup_fit",
                "weights": {
                    "stability": 0.30,
                    "revenue_capacity": 0.25,
                    "competition_balance": 0.20,
                    "market_momentum": 0.15,
                    "demand_capacity": 0.10,
                },
                "caution": "relative decision support score; not a success guarantee or causal estimate",
            },
        },
        "industries": industries,
        "dongs": dongs,
        "records": records,
        "sources": {
            "population": sources["population"].name,
            "sales": sources["sales"].name,
            "stores": sources["stores"].name,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        json.dump(output, target, ensure_ascii=False, separators=(",", ":"))
    print(f"Built {len(records):,} dashboard records for {latest}.")


if __name__ == "__main__":
    main()
