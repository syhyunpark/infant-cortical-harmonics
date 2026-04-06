#!/usr/bin/env python3
# simulate_recoverability_basis_mismatch.py
# Recoverability under matched infant analysis basis vs adult-derived mismatched basis.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate recoverability under basis mismatch"
    )
    parser.add_argument("--gendir", required=True, help="Generation dictionary directory")
    parser.add_argument("--analysisdir", required=True, help="Analysis dictionary directory")
    parser.add_argument("--analysis-mode", required=True, choices=["matched", "adult_mismatch"])

    parser.add_argument("--target-k", type=int, default=50, help="Modes per hemisphere in saved dictionaries")
    parser.add_argument("--analysis-k", type=int, default=10, help="Modes per hemisphere used in the simulation")

    parser.add_argument("--surface", default="white")
    parser.add_argument("--combine", default="block")
    parser.add_argument("--montage-kind", default="template1020")

    parser.add_argument("--snr-db-list", default="20,10,0")
    parser.add_argument("--prior-kind", choices=["equal", "decay"], default="equal")
    parser.add_argument("--decay-p", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.7)

    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--n-time", type=int, default=600)
    parser.add_argument("--pair-hemispheres", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--outdir", required=True)

    return parser.parse_args()


def infer_scalar(x, key, default=None):
    if key not in x:
        return default

    arr = x[key]
    if getattr(arr, "shape", ()) == ():
        return arr.item()

    if len(arr) == 1:
        value = arr[0]
        return value.item() if hasattr(value, "item") else value

    return arr


def age_to_months(age: str) -> float:
    age = str(age).strip().lower()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(wk|mo|yr)", age)
    if m is None:
        return float("inf")

    value = float(m.group(1))
    unit = m.group(2)

    if unit == "wk":
        return value / 4.34524
    if unit == "mo":
        return value
    if unit == "yr":
        return value * 12.0

    return float("inf")


def load_rows(dictdir: Path, target_k: int, surface: str, combine: str, montage_kind: str):
    rows = {}

    for path in sorted(dictdir.glob("*.npz")):
        x = np.load(path, allow_pickle=True)

        K = int(np.array(x["K"]).ravel()[0]) if "K" in x else None
        surf = infer_scalar(x, "surface")
        comb = infer_scalar(x, "combine")
        mont = infer_scalar(x, "montage_kind")

        if K != target_k:
            continue
        if surf is not None and surf != surface:
            continue
        if comb is not None and comb != combine:
            continue
        if mont is not None and mont != montage_kind:
            continue

        age = infer_scalar(x, "age", path.stem)
        rows[age] = {
            "file": path,
            "age": age,
            "subject": infer_scalar(x, "subject", path.stem),
            "G": x["G"].astype(float),
            "D": x["D"].astype(float),
            "K": K,
        }

    if not rows:
        raise RuntimeError(f"No matching rows found in {dictdir}")

    return rows


def select_columns(K_total: int, K_use: int) -> np.ndarray:
    lh = np.arange(0, K_use)
    rh = np.arange(K_total, K_total + K_use)
    return np.concatenate([lh, rh])


def make_var_vector(K_use: int, prior_kind: str, decay_p: float) -> np.ndarray:
    idx = np.arange(1, K_use + 1, dtype=float)

    if prior_kind == "equal":
        return np.ones(K_use, dtype=float)

    return idx ** (-decay_p)


def make_block_var(K_use: int, prior_kind: str, decay_p: float) -> np.ndarray:
    v = make_var_vector(K_use, prior_kind, decay_p)
    return np.concatenate([v, v])


def simulate_ar1_coeffs(
    n_time: int,
    var_vec: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    P = len(var_vec)
    X = np.zeros((n_time, P), dtype=float)

    sd = np.sqrt(var_vec)
    X[0] = rng.normal(scale=sd, size=P)

    innov_sd = np.sqrt(1.0 - rho**2) * sd
    for t in range(1, n_time):
        X[t] = rho * X[t - 1] + rng.normal(scale=innov_sd, size=P)

    return X


def expected_signal_power(D: np.ndarray, var_vec: np.ndarray) -> float:
    cov = np.diag(var_vec)
    signal_cov = D @ cov @ D.T
    return float(np.mean(np.diag(signal_cov)))


def posterior_mean(Y: np.ndarray, D: np.ndarray, var_vec: np.ndarray, sigma2: float) -> np.ndarray:
    Sigma = np.diag(var_vec)
    A = D @ Sigma @ D.T + sigma2 * np.eye(D.shape[0])
    Kmat = Sigma @ D.T @ np.linalg.inv(A)
    return (Kmat @ Y.T).T


def corr_cols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        x = X[:, j] - X[:, j].mean()
        y = Y[:, j] - Y[:, j].mean()

        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)

        if nx < 1e-12 or ny < 1e-12:
            out[j] = np.nan
        else:
            out[j] = np.dot(x, y) / (nx * ny)

    return out


def pair_corrs(corrs: np.ndarray, K_use: int) -> np.ndarray:
    return 0.5 * (corrs[:K_use] + corrs[K_use:])


def largest_above_threshold(x: np.ndarray, threshold: float) -> int:
    idx = np.where(x >= threshold)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1] + 1)


def contiguous_from_start(x: np.ndarray, threshold: float) -> int:
    n = 0
    for value in x:
        if value >= threshold:
            n += 1
        else:
            break
    return n


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    g_rows = load_rows(
        Path(args.gendir).expanduser().resolve(),
        args.target_k,
        args.surface,
        args.combine,
        args.montage_kind,
    )
    a_rows = load_rows(
        Path(args.analysisdir).expanduser().resolve(),
        args.target_k,
        args.surface,
        args.combine,
        args.montage_kind,
    )

    common_ages = sorted(set(g_rows) & set(a_rows), key=age_to_months)
    if not common_ages:
        raise RuntimeError("No common ages between generation and analysis directories.")

    snr_list = [float(x.strip()) for x in args.snr_db_list.split(",") if x.strip()]

    summary_csv = outdir / "recoverability_summary.csv"
    raw_csv = outdir / "recoverability_raw_modecorr.csv"

    with summary_csv.open("w", newline="") as fsum, raw_csv.open("w", newline="") as fraw:
        sum_writer = csv.writer(fsum)
        raw_writer = csv.writer(fraw)

        sum_writer.writerow(
            [
                "analysis_mode",
                "age",
                "subject",
                "snr_db",
                "analysis_k",
                "prior_kind",
                "signal_power_expected",
                "sigma2",
                "mean_corr_first5",
                "mean_corr_first10",
                "mean_corr_all",
                "Krec_largest",
                "Krec_contig",
            ]
        )

        raw_writer.writerow(
            [
                "analysis_mode",
                "age",
                "subject",
                "snr_db",
                "analysis_k",
                "prior_kind",
                "mode_order",
                "paired_corr",
            ]
        )

        for age in common_ages:
            g = g_rows[age]
            a = a_rows[age]

            idx = select_columns(g["K"], args.analysis_k)

            Dg = g["D"][:, idx]
            Da = a["D"][:, idx]

            var_vec = make_block_var(args.analysis_k, args.prior_kind, args.decay_p)
            sig_pow = expected_signal_power(Dg, var_vec)

            for snr_db in snr_list:
                sigma2 = sig_pow / (10.0 ** (snr_db / 10.0))

                all_true = []
                all_hat = []

                for _ in range(args.n_reps):
                    Wtrue = simulate_ar1_coeffs(args.n_time, var_vec, args.rho, rng)
                    Yclean = Wtrue @ Dg.T
                    Y = Yclean + rng.normal(scale=np.sqrt(sigma2), size=Yclean.shape)

                    What = posterior_mean(Y, Da, var_vec, sigma2)

                    all_true.append(Wtrue)
                    all_hat.append(What)

                all_true = np.vstack(all_true)
                all_hat = np.vstack(all_hat)

                corrs = corr_cols(all_true, all_hat)
                paired = pair_corrs(corrs, args.analysis_k) if args.pair_hemispheres else corrs

                mean_corr_first5 = float(np.nanmean(paired[: min(5, len(paired))]))
                mean_corr_first10 = float(np.nanmean(paired[: min(10, len(paired))]))
                mean_corr_all = float(np.nanmean(paired))

                Krec_largest = largest_above_threshold(paired, args.threshold)
                Krec_contig = contiguous_from_start(paired, args.threshold)

                sum_writer.writerow(
                    [
                        args.analysis_mode,
                        age,
                        g["subject"],
                        snr_db,
                        args.analysis_k,
                        args.prior_kind,
                        sig_pow,
                        sigma2,
                        mean_corr_first5,
                        mean_corr_first10,
                        mean_corr_all,
                        Krec_largest,
                        Krec_contig,
                    ]
                )

                for k, value in enumerate(paired, start=1):
                    raw_writer.writerow(
                        [
                            args.analysis_mode,
                            age,
                            g["subject"],
                            snr_db,
                            args.analysis_k,
                            args.prior_kind,
                            k,
                            value,
                        ]
                    )

                print(
                    f"{args.analysis_mode:>14} | {age:>5} | SNR={snr_db:>4.0f} dB | "
                    f"mean first10={mean_corr_first10:.3f} | Krec={Krec_largest}"
                )

    print(f"\nSaved -> {summary_csv}")
    print(f"Saved -> {raw_csv}")


if __name__ == "__main__":
    main()