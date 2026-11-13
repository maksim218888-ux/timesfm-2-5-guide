#!/usr/bin/env python3
"""Reproducible TimesFM 2.5 evaluation on NOAA Mauna Loa monthly CO2.

Produces a holdout forecast, a seasonal-naive baseline, a rolling-origin
backtest, MAE/WAPE/q10-q90 coverage, a future forecast, CSV/JSON files, and PNG
plots. Optional XReg covariates use only future-known calendar month and a
deterministic time index.
"""

from __future__ import annotations

import argparse
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

HORIZON = 12
SEASON_LENGTH = 12
Q10_COL = 1
Q50_COL = 5
Q90_COL = 9
MODEL_ID = "google/timesfm-2.5-200m-pytorch"
MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "noaa_co2_snapshot.csv",
    )
    parser.add_argument("--output", type=Path, default=here / "output")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--with-covariates", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, comment="#")
    required = {"year", "month", "average"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.loc[:, ["year", "month", "average"]].copy()
    df["date"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}, errors="raise"
    )
    df["y"] = pd.to_numeric(df["average"], errors="coerce")
    df.loc[df["y"] < 0, "y"] = np.nan
    df = df.loc[:, ["date", "y"]].sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate timestamps found")
    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    if not df["date"].equals(pd.Series(expected)):
        raise ValueError("Dates are not a complete regular monthly sequence")
    if df["y"].isna().any():
        raise ValueError("Target contains missing/negative sentinel values")
    if len(df) < 10 * SEASON_LENGTH:
        raise ValueError("Need at least 10 years of monthly history")
    return df


def build_model(*, local_files_only: bool, return_backcast: bool = False):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_compile=False,
        local_files_only=local_files_only,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=512,
            max_horizon=128,
            per_core_batch_size=1,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=return_backcast,
        )
    )
    return model


def seasonal_naive(train: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    if len(train) < SEASON_LENGTH:
        raise ValueError("Seasonal naive requires at least 12 observations")
    return np.resize(train[-SEASON_LENGTH:], horizon).astype(np.float64)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape=} {predicted.shape=}")
    denom = float(np.abs(actual).sum())
    if denom == 0:
        raise ZeroDivisionError("WAPE is undefined because sum(abs(actual)) is zero")
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "wape_pct": float(100.0 * np.abs(actual - predicted).sum() / denom),
    }


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Coverage arrays must have identical shapes")
    if np.any(lower > upper):
        raise ValueError("Crossed interval bounds remain after post-processing")
    return float(np.mean((actual >= lower) & (actual <= upper)))


def forecast_batch(model, contexts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    clean = [np.asarray(x, dtype=np.float32) for x in contexts]
    if any(x.ndim != 1 or len(x) == 0 for x in clean):
        raise ValueError("Each TimesFM input must be a non-empty 1-D array")
    point, quantiles = model.forecast(horizon=HORIZON, inputs=clean)
    if point.shape != (len(clean), HORIZON):
        raise RuntimeError(f"Unexpected point shape: {point.shape}")
    if quantiles.shape != (len(clean), HORIZON, 10):
        raise RuntimeError(f"Unexpected quantile shape: {quantiles.shape}")
    if not np.allclose(point, quantiles[:, :, Q50_COL], atol=1e-5, rtol=1e-5):
        raise RuntimeError("Point forecast does not match q50 column")
    return point, quantiles


def rolling_backtest(
    model, values: np.ndarray, dates: pd.Series, folds: int
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if folds < 2:
        raise ValueError("Use at least two rolling folds")
    if len(values) <= (folds + 1) * HORIZON:
        raise ValueError("Not enough history for requested rolling folds")

    origins = [len(values) - HORIZON * (folds - i) for i in range(folds)]
    contexts = [values[:origin] for origin in origins]
    point, quantiles = forecast_batch(model, contexts)

    rows: list[dict[str, object]] = []
    all_actual, all_tfm, all_naive, all_q10, all_q90 = [], [], [], [], []
    for fold, origin in enumerate(origins, start=1):
        actual = values[origin : origin + HORIZON]
        naive = seasonal_naive(values[:origin])
        q10 = quantiles[fold - 1, :, Q10_COL]
        q90 = quantiles[fold - 1, :, Q90_COL]
        for step in range(HORIZON):
            rows.append(
                {
                    "fold": fold,
                    "origin_date": dates.iloc[origin - 1].date().isoformat(),
                    "target_date": dates.iloc[origin + step].date().isoformat(),
                    "actual": float(actual[step]),
                    "timesfm": float(point[fold - 1, step]),
                    "naive": float(naive[step]),
                    "q10": float(q10[step]),
                    "q90": float(q90[step]),
                }
            )
        all_actual.append(actual)
        all_tfm.append(point[fold - 1])
        all_naive.append(naive)
        all_q10.append(q10)
        all_q90.append(q90)

    actual = np.concatenate(all_actual)
    tfm = np.concatenate(all_tfm)
    naive = np.concatenate(all_naive)
    q10 = np.concatenate(all_q10)
    q90 = np.concatenate(all_q90)
    summary = {
        "timesfm": metrics(actual, tfm),
        "seasonal_naive": metrics(actual, naive),
        "timesfm_q10_q90": {
            "nominal_coverage": 0.80,
            "empirical_coverage": coverage(actual, q10, q90),
            "mean_width": float(np.mean(q90 - q10)),
        },
    }
    return pd.DataFrame(rows), summary


def future_known_covariates(context_dates: pd.Series, horizon: int) -> tuple[dict, dict]:
    future_dates = pd.date_range(
        context_dates.iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    all_dates = pd.DatetimeIndex(context_dates.tolist() + future_dates.tolist())
    time_index = np.arange(len(all_dates), dtype=np.float32)
    month = all_dates.month.astype(np.int32)
    return {"time_index": [time_index]}, {"month": [month]}


def covariate_forecast(model, values: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    dynamic_num, dynamic_cat = future_known_covariates(dates, HORIZON)
    point, quantiles = model.forecast_with_covariates(
        inputs=[values.astype(np.float32)],
        dynamic_numerical_covariates=dynamic_num,
        dynamic_categorical_covariates=dynamic_cat,
        xreg_mode="xreg + timesfm",
        ridge=1e-3,
        force_on_cpu=True,
    )
    point_arr = np.asarray(point[0])
    quantile_arr = np.asarray(quantiles[0])
    future_dates = pd.date_range(
        dates.iloc[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS"
    )
    return pd.DataFrame(
        {
            "date": future_dates,
            "point": point_arr,
            "q10": quantile_arr[:, Q10_COL],
            "q50": quantile_arr[:, Q50_COL],
            "q90": quantile_arr[:, Q90_COL],
        }
    )


def plot_holdout(
    df: pd.DataFrame,
    holdout: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    history = df.iloc[-72:]
    ax.plot(history["date"], history["y"], label="Actual history", color="black")
    ax.plot(holdout["date"], holdout["timesfm"], label="TimesFM q50", color="#1565c0")
    ax.plot(holdout["date"], holdout["naive"], label="Seasonal naive", color="#ef6c00")
    ax.fill_between(
        holdout["date"], holdout["q10"], holdout["q90"], alpha=0.2, color="#1565c0", label="q10–q90"
    )
    ax.axvline(holdout["date"].iloc[0], color="gray", linestyle="--", linewidth=1)
    ax.set(title="TimesFM 2.5 — honest 12-month holdout", ylabel="CO₂ monthly mean (ppm)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_future(df: pd.DataFrame, future: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    history = df.iloc[-72:]
    ax.plot(history["date"], history["y"], label="Observed", color="black")
    ax.plot(future["date"], future["point"], label="TimesFM q50", color="#2e7d32")
    ax.fill_between(
        future["date"], future["q10"], future["q90"], alpha=0.2, color="#2e7d32", label="q10–q90"
    )
    ax.axvline(future["date"].iloc[0], color="gray", linestyle="--", linewidth=1)
    ax.set(title="TimesFM 2.5 — future 12-month forecast", ylabel="CO₂ monthly mean (ppm)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    df = load_data(args.data)
    values = df["y"].to_numpy(dtype=np.float32)

    model = build_model(local_files_only=args.local_files_only)

    train, actual = values[:-HORIZON], values[-HORIZON:]
    holdout_dates = df["date"].iloc[-HORIZON:].reset_index(drop=True)
    point, quantiles = forecast_batch(model, [train])
    naive = seasonal_naive(train)
    holdout = pd.DataFrame(
        {
            "date": holdout_dates,
            "actual": actual,
            "timesfm": point[0],
            "naive": naive,
            "q10": quantiles[0, :, Q10_COL],
            "q50": quantiles[0, :, Q50_COL],
            "q90": quantiles[0, :, Q90_COL],
        }
    )
    holdout_metrics = {
        "timesfm": metrics(actual, point[0]),
        "seasonal_naive": metrics(actual, naive),
        "timesfm_q10_q90": {
            "nominal_coverage": 0.80,
            "empirical_coverage": coverage(
                actual, quantiles[0, :, Q10_COL], quantiles[0, :, Q90_COL]
            ),
            "mean_width": float(
                np.mean(quantiles[0, :, Q90_COL] - quantiles[0, :, Q10_COL])
            ),
        },
    }

    rolling, rolling_metrics = rolling_backtest(model, values, df["date"], args.folds)

    future_point, future_quantiles = forecast_batch(model, [values])
    future_dates = pd.date_range(
        df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS"
    )
    future = pd.DataFrame(
        {
            "date": future_dates,
            "point": future_point[0],
            "q10": future_quantiles[0, :, Q10_COL],
            "q50": future_quantiles[0, :, Q50_COL],
            "q90": future_quantiles[0, :, Q90_COL],
        }
    )

    covariate_path = None
    if args.with_covariates:
        # XReg requires return_backcast=True, so recompile the already-loaded model.
        model.compile(
            timesfm.ForecastConfig(
                max_context=512,
                max_horizon=128,
                per_core_batch_size=1,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                return_backcast=True,
            )
        )
        covariate = covariate_forecast(model, values, df["date"])
        covariate_path = args.output / "future_forecast_with_covariates.csv"
        covariate.to_csv(covariate_path, index=False)

    holdout.to_csv(args.output / "holdout_forecast.csv", index=False)
    rolling.to_csv(args.output / "rolling_backtest.csv", index=False)
    future.to_csv(args.output / "future_forecast.csv", index=False)
    plot_holdout(df, holdout, args.output / "holdout_forecast.png")
    plot_future(df, future, args.output / "future_forecast.png")

    report = {
        "run": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "timesfm": importlib.metadata.version("timesfm"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "local_files_only": args.local_files_only,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
        "data": {
            "path": str(args.data),
            "rows": len(df),
            "start": df["date"].iloc[0].date().isoformat(),
            "end": df["date"].iloc[-1].date().isoformat(),
            "target": "average monthly atmospheric CO2, ppm",
        },
        "design": {
            "holdout_horizon": HORIZON,
            "rolling_folds": args.folds,
            "rolling_horizon": HORIZON,
            "baseline": "seasonal naive, period=12",
            "interval": "q10-q90, nominal coverage=0.80",
        },
        "holdout": holdout_metrics,
        "rolling": rolling_metrics,
        "covariate_output": str(covariate_path) if covariate_path else None,
    }
    with (args.output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, ZeroDivisionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
