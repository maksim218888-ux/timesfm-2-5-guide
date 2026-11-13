#!/usr/bin/env python3
"""Agent-ready TimesFM 2.5 workflow for one regular numeric business series.

Input: CSV with a date column and one numeric target column.
Output: data-quality report, holdout and rolling backtests, baseline comparison,
future point/range forecast, charts, run manifest and a plain-Russian decision memo.
The script produces decision support only and never performs business actions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import timesfm

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
Q10_COL, Q50_COL, Q90_COL = 1, 5, 9


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=here / "business_series.csv")
    p.add_argument("--output", type=Path, default=here / "output")
    p.add_argument("--date-col", default="timestamp")
    p.add_argument("--target-col", default="target")
    p.add_argument("--series-col", default="series_id")
    p.add_argument("--series-id", default=None)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--season-length", type=int, default=7)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--purpose", default="inventory_planning")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_series(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    if not args.data.is_file():
        raise FileNotFoundError(f"CSV not found: {args.data}")
    raw = pd.read_csv(args.data, comment="#")

    # Backward-compatible NOAA demo mapping.
    if {"year", "month", "average"}.issubset(raw.columns) and args.date_col not in raw:
        raw["timestamp"] = pd.to_datetime(
            {"year": raw["year"], "month": raw["month"], "day": 1}, errors="raise"
        )
        raw["target"] = raw["average"]
        raw["series_id"] = "noaa_mauna_loa_co2"
        date_col, target_col, series_col = "timestamp", "target", "series_id"
    else:
        date_col, target_col, series_col = args.date_col, args.target_col, args.series_col

    missing = {date_col, target_col}.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    if series_col in raw.columns:
        ids = raw[series_col].dropna().astype(str).unique().tolist()
        selected = args.series_id or (ids[0] if len(ids) == 1 else None)
        if selected is None:
            raise ValueError(f"CSV contains multiple series. Pass --series-id. Found: {ids[:10]}")
        raw = raw.loc[raw[series_col].astype(str) == str(selected)].copy()
        series_id = str(selected)
    else:
        series_id = args.series_id or "series_1"

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="raise", utc=False),
        "y": pd.to_numeric(raw[target_col], errors="coerce"),
    }).sort_values("date").reset_index(drop=True)

    duplicate_count = int(df["date"].duplicated().sum())
    missing_target_count = int(df["y"].isna().sum())
    if duplicate_count:
        raise ValueError(f"Duplicate dates: {duplicate_count}")
    if missing_target_count:
        raise ValueError(f"Missing/non-numeric target values: {missing_target_count}")
    if not np.isfinite(df["y"].to_numpy(dtype=float)).all():
        raise ValueError("Target contains infinite values")
    if len(df) < max(4 * args.season_length, (args.folds + 1) * args.horizon + args.season_length):
        raise ValueError(
            "Not enough history. Need at least four seasonal cycles and enough rows for "
            f"{args.folds} rolling checks; received {len(df)} rows."
        )

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        deltas = df["date"].diff().dropna()
        raise ValueError(
            "Dates are not regular. Aggregate or fill the calendar explicitly before forecasting. "
            f"Observed interval examples: {deltas.value_counts().head(3).to_dict()}"
        )

    quality = {
        "ok": True,
        "series_id": series_id,
        "rows": int(len(df)),
        "start": df["date"].iloc[0].isoformat(),
        "end": df["date"].iloc[-1].isoformat(),
        "inferred_frequency": inferred,
        "date_column": date_col,
        "target_column": target_col,
        "duplicate_dates": duplicate_count,
        "missing_target_values": missing_target_count,
        "numeric_target": True,
        "minimum_cycles_check": "pass",
        "warnings": [],
    }
    return df, quality


def build_model(local_files_only: bool):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_compile=False,
        local_files_only=local_files_only,
    )
    model.compile(timesfm.ForecastConfig(
        max_context=512, max_horizon=128, per_core_batch_size=1,
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=False,
        fix_quantile_crossing=True,
    ))
    return model


def forecast_batch(model, contexts: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if horizon < 1 or horizon > 128:
        raise ValueError("--horizon must be between 1 and 128")
    clean = [np.asarray(x, dtype=np.float32) for x in contexts]
    point, quantiles = model.forecast(horizon=horizon, inputs=clean)
    if point.shape != (len(clean), horizon) or quantiles.shape != (len(clean), horizon, 10):
        raise RuntimeError(f"Unexpected TimesFM shapes: point={point.shape}, quantiles={quantiles.shape}")
    if not np.allclose(point, quantiles[:, :, Q50_COL], atol=1e-5, rtol=1e-5):
        raise RuntimeError("Point forecast does not match q50")
    if not (np.isfinite(point).all() and np.isfinite(quantiles).all()):
        raise RuntimeError("Forecast contains non-finite values")
    if np.any(quantiles[:, :, Q10_COL] > quantiles[:, :, Q50_COL]) or np.any(quantiles[:, :, Q50_COL] > quantiles[:, :, Q90_COL]):
        raise RuntimeError("Quantile ordering failed")
    return point, quantiles


def seasonal_naive(train: np.ndarray, horizon: int, season: int) -> np.ndarray:
    if len(train) < season:
        raise ValueError("History is shorter than the baseline season")
    return np.resize(train[-season:], horizon).astype(float)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mae = float(np.mean(np.abs(actual - pred)))
    denom = float(np.abs(actual).sum())
    return {"mae": mae, "wape_pct": None if denom == 0 else float(100 * np.abs(actual - pred).sum() / denom)}


def make_future_dates(last_date: pd.Timestamp, inferred_freq: str, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date, periods=horizon + 1, freq=inferred_freq)[1:]


def rolling_backtest(model, values, dates, horizon, season, folds):
    origins = [len(values) - horizon * (folds - i) for i in range(folds)]
    if min(origins) < season:
        raise ValueError("Not enough history for requested folds, horizon and season")
    point, quantiles = forecast_batch(model, [values[:o] for o in origins], horizon)
    rows, actual_all, tfm_all, base_all, low_all, high_all = [], [], [], [], [], []
    for i, origin in enumerate(origins):
        actual = values[origin:origin+horizon]
        baseline = seasonal_naive(values[:origin], horizon, season)
        low, high = quantiles[i, :, Q10_COL], quantiles[i, :, Q90_COL]
        for step in range(horizon):
            rows.append({
                "fold": i + 1, "origin_date": dates.iloc[origin-1].date().isoformat(),
                "target_date": dates.iloc[origin+step].date().isoformat(),
                "actual": float(actual[step]), "timesfm_q50": float(point[i, step]),
                "baseline": float(baseline[step]), "q10": float(low[step]), "q90": float(high[step]),
            })
        actual_all.append(actual); tfm_all.append(point[i]); base_all.append(baseline); low_all.append(low); high_all.append(high)
    actual = np.concatenate(actual_all); tfm = np.concatenate(tfm_all); base = np.concatenate(base_all)
    low = np.concatenate(low_all); high = np.concatenate(high_all)
    return pd.DataFrame(rows), {
        "timesfm": metrics(actual, tfm), "baseline": metrics(actual, base),
        "q10_q90": {"nominal_coverage": 0.8, "empirical_coverage": float(np.mean((actual >= low) & (actual <= high))), "mean_width": float(np.mean(high-low))},
    }


def write_plots(df, holdout, future, output):
    history = df.iloc[-min(len(df), 120):]
    fig, ax = plt.subplots(figsize=(11, 5.5)); ax.plot(history.date, history.y, color="black", label="История")
    ax.plot(holdout.date, holdout.timesfm_q50, color="#4d7c0f", label="TimesFM q50")
    ax.plot(holdout.date, holdout.baseline, color="#ff6900", label="Baseline")
    ax.fill_between(holdout.date, holdout.q10, holdout.q90, color="#a3e635", alpha=.25, label="q10–q90")
    ax.axvline(holdout.date.iloc[0], color="gray", linestyle="--"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output/"holdout_comparison.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.5)); ax.plot(history.date, history.y, color="black", label="История")
    ax.plot(future.date, future.q50, color="#4d7c0f", label="Прогноз q50")
    ax.fill_between(future.date, future.q10, future.q90, color="#a3e635", alpha=.25, label="Диапазон q10–q90")
    ax.axvline(future.date.iloc[0], color="gray", linestyle="--"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output/"future_forecast.png", dpi=160); plt.close(fig)


def decision_memo(args, quality, rolling_metrics) -> str:
    tfm = rolling_metrics["timesfm"]["mae"]
    base = rolling_metrics["baseline"]["mae"]
    coverage = rolling_metrics["q10_q90"]["empirical_coverage"]
    if tfm < base:
        verdict = "TimesFM точнее простого baseline на этой исторической проверке. Его можно использовать как дополнительный сигнал для ограниченного пилота."
    else:
        verdict = "TimesFM не обошёл простой baseline на этой исторической проверке. Не усложняйте процесс: оставьте baseline или пересмотрите данные и постановку задачи."
    return f"""# Служебная записка для владельца\n\n## Что проверяли\nРяд: **{quality['series_id']}**. Цель решения: **{args.purpose}**. Горизонт: **{args.horizon}** наблюдений.\n\n## Что получилось\n- MAE TimesFM: **{tfm:.4f}**\n- MAE baseline: **{base:.4f}**\n- Покрытие диапазона q10–q90: **{coverage:.1%}**\n\n## Вывод\n{verdict}\n\n## Как читать прогноз\n- q50: центральный сценарий модели;\n- q10–q90: диапазон сценариев, а не гарантия;\n- baseline: простое правило, с которым обязана сравниться сложная модель.\n\n## Решение владельца\nПрогноз не заказывает товар, не меняет смены, не увеличивает рекламу и не переводит деньги. Владелец выбирает: **не использовать / провести пилот / использовать как дополнительный сигнал**.\n"""


def main() -> int:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    df, quality = load_series(args); values = df.y.to_numpy(np.float32)
    (args.output/"data_quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    model = build_model(args.local_files_only)
    train, actual = values[:-args.horizon], values[-args.horizon:]
    point, quantiles = forecast_batch(model, [train], args.horizon)
    baseline = seasonal_naive(train, args.horizon, args.season_length)
    holdout = pd.DataFrame({"date": df.date.iloc[-args.horizon:].reset_index(drop=True), "actual": actual, "timesfm_q50": point[0], "baseline": baseline, "q10": quantiles[0,:,Q10_COL], "q50": quantiles[0,:,Q50_COL], "q90": quantiles[0,:,Q90_COL]})
    rolling, rolling_metrics = rolling_backtest(model, values, df.date, args.horizon, args.season_length, args.folds)
    future_point, future_q = forecast_batch(model, [values], args.horizon)
    future = pd.DataFrame({"date": make_future_dates(df.date.iloc[-1], quality["inferred_frequency"], args.horizon), "q10": future_q[0,:,Q10_COL], "q50": future_q[0,:,Q50_COL], "q90": future_q[0,:,Q90_COL], "point": future_point[0]})
    holdout.to_csv(args.output/"holdout_comparison.csv", index=False); rolling.to_csv(args.output/"rolling_backtest.csv", index=False); future.to_csv(args.output/"future_forecast.csv", index=False)
    write_plots(df, holdout, future, args.output)
    holdout_metrics = {"timesfm": metrics(actual, point[0]), "baseline": metrics(actual, baseline), "q10_q90": {"nominal_coverage": .8, "empirical_coverage": float(np.mean((actual >= quantiles[0,:,Q10_COL]) & (actual <= quantiles[0,:,Q90_COL]))), "mean_width": float(np.mean(quantiles[0,:,Q90_COL]-quantiles[0,:,Q10_COL]))}}
    report = {"quality": quality, "design": {"horizon": args.horizon, "season_length": args.season_length, "folds": args.folds, "purpose": args.purpose}, "holdout": holdout_metrics, "rolling": rolling_metrics, "decision_support_only": True}
    (args.output/"quality_and_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (args.output/"DECISION-MEMO.md").write_text(decision_memo(args, quality, rolling_metrics), encoding="utf-8")
    manifest = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "timesfm_package": importlib.metadata.version("timesfm"), "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "input_csv": str(args.data), "input_sha256": sha256(args.data), "elapsed_seconds": round(time.perf_counter()-started, 3), "outputs": sorted(p.name for p in args.output.iterdir()), "no_business_action_performed": True}
    (args.output/"run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "rolling": rolling_metrics, "decision_memo": str(args.output/"DECISION-MEMO.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2) from exc
