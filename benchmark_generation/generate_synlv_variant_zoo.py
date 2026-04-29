# -*- coding: utf-8 -*-
"""SynLV primary and sensitivity-branch generation utilities.

This module provides generator functions used by the public v1.0 release for
MNAR-correlated and transfer scenarios, plus source-reproducible sensitivity
branches used in appendix analyses. The hosted v1.0 benchmark payload contains
only the primary scenarios documented in the scenario registry.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class GenCfg:
    n_patients: int = 8000
    n_cont: int = 12
    n_cat: int = 4
    d_latent: int = 4
    t_max_days: int = 9000
    dt_days: int = 10
    min_visits: int = 3
    max_visits: int = 6
    mask_pool_size: int = 6
    lam0: float = 1e-4
    near_event_window_days: int = 600
    min_gap_bins: int = 3
    test_frac: float = 0.20
    train_frac: float = 0.65
    split_seed: int = 1991
    mask_categorical: int = 0
    censor_lam0: float = 5.0e-5
    censor_risk_gamma: float = 0.9
    censor_min_days: int = 120
    # MNAR-correlated scenario controls (Gaussian-copula Bernoulli).
    mnar_corr_rho: float = 0.55
    mnar_corr_structure: str = "exchangeable"  # exchangeable | block
    mnar_corr_block_size: int = 6
    mnar_corr_state_dependent: int = 0
    mnar_corr_state_scale: float = 0.10
    # Nonlinearity ladder control (0.0 -> base-like, 1.0 -> nonlinear_obs reference configuration).
    nonlinear_alpha: float = 1.0


# Train/test transfer scenarios (train/val from reference, test from the named control scenario).
TRANSFER_SCENARIO_MAP = {
    "scenario_VISITSHIFT_TRANSFER": ("scenario_A", "scenario_VISITSHIFT"),
    "scenario_MISMATCH_TRANSFER": ("scenario_A", "scenario_MISMATCH"),
}

# MNAR strength sensitivity scenarios.
MNAR_ALPHA_SCENARIO_MAP = {
    "scenario_MNAR_ALPHA_0p0": 0.0,
    "scenario_MNAR_ALPHA_0p4": 0.4,
    "scenario_MNAR_ALPHA_0p8": 0.8,
    "scenario_MNAR_ALPHA_1p2": 1.2,
    "scenario_MNAR_ALPHA_1p6": 1.6,
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _stable_sigmoid_scalar(x: float) -> float:
    """Numerically stable scalar sigmoid used in latent branch transitions."""
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(-x))


def _build_corr_chol(n_features: int, rho: float, structure: str = "exchangeable", block_size: int = 6):
    rho = float(np.clip(rho, 0.0, 0.95))
    if structure == "block":
        corr = np.eye(n_features, dtype=float)
        block = max(1, min(int(block_size), n_features))
        for s in range(0, n_features, block):
            e = min(n_features, s + block)
            corr[s:e, s:e] = rho
            np.fill_diagonal(corr[s:e, s:e], 1.0)
    else:
        corr = np.full((n_features, n_features), rho, dtype=float)
        np.fill_diagonal(corr, 1.0)
    jitter = 1e-8
    return np.linalg.cholesky(corr + jitter * np.eye(n_features, dtype=float))


def _corr_blocks_from_size(n_features: int, block_size: int) -> list[list[int]]:
    """Deterministic contiguous block specification used for block-structured MNAR-corr."""
    n_features = int(max(1, n_features))
    block = max(1, min(int(block_size), n_features))
    blocks: list[list[int]] = []
    for s in range(0, n_features, block):
        e = min(n_features, s + block)
        blocks.append(list(range(s, e)))
    return blocks


def _sample_correlated_bernoulli_gaussian_copula(rng, p_miss: np.ndarray, chol: np.ndarray):
    # Preserve channel-wise marginals (p_j) while inducing positive correlation.
    p = np.clip(np.asarray(p_miss, dtype=float), 1e-6, 1.0 - 1e-6)
    z = chol @ rng.standard_normal(size=p.shape[0])
    q = norm.ppf(p)
    return z < q


def _latent_at_t(
    t,
    z0,
    rates,
    variant,
    rng,
    patient_cache,
):
    # Baseline closed-form exponential decay.
    z = z0 * np.exp(-rates * t)

    if variant == "acute_hetero":
        # Heterogeneous subpopulations with acute/non-monotone trajectories.
        # This variant changes latent dynamics only; the observation map, hazard decoder, and stress protocol are unchanged.
        if "acute_group" not in patient_cache:
            mix_probs = np.array([0.40, 0.35, 0.25], dtype=float)
            acute_group = int(rng.choice(3, p=mix_probs))
            te = max(1.0, float(patient_cache.get("t_event_ref", 1.0)))
            d = int(rates.shape[0])
            acute_dir = rng.normal(0.0, 1.0, size=d)
            acute_dir = acute_dir / (np.linalg.norm(acute_dir) + 1e-12)
            acute_mix = rng.normal(0.0, 0.12, size=(d, d))
            tau_on = float(rng.uniform(0.20, 0.55) * te)
            tau_off = float(rng.uniform(0.65, 0.92) * te)
            if tau_off <= tau_on:
                tau_off = tau_on + max(1.0, 0.05 * te)
            patient_cache["acute_group"] = acute_group
            patient_cache["acute_group_name"] = ["smooth_baseline", "late_acute_worsening", "partial_recovery"][acute_group]
            patient_cache["acute_dir"] = acute_dir
            patient_cache["acute_mix"] = acute_mix
            patient_cache["acute_tau_on"] = tau_on
            patient_cache["acute_tau_off"] = tau_off
            patient_cache["acute_width"] = float(max(1.0, rng.uniform(0.03, 0.08) * te))
            patient_cache["acute_amp"] = float(rng.uniform(0.75, 1.90))
            patient_cache["acute_phase"] = float(rng.uniform(0.0, 2.0 * np.pi))
            patient_cache["acute_te"] = te

        group = int(patient_cache["acute_group"])
        te = max(1.0, float(patient_cache.get("acute_te", patient_cache.get("t_event_ref", 1.0))))
        tau_on = float(patient_cache["acute_tau_on"])
        tau_off = float(patient_cache["acute_tau_off"])
        width = float(patient_cache["acute_width"])
        amp = float(patient_cache["acute_amp"])
        acute_dir = np.asarray(patient_cache["acute_dir"], dtype=float)
        acute_mix = np.asarray(patient_cache["acute_mix"], dtype=float)

        # Subgroup 0: smoother baseline with mild low-frequency modulation.
        if group == 0:
            osc = 0.10 * np.sin((2.0 * np.pi * t / te) + float(patient_cache["acute_phase"]))
            z = z + osc * np.tanh(acute_mix @ z)
        # Subgroup 1: late acute worsening (abrupt deterioration near end of follow-up).
        elif group == 1:
            jump = amp * _stable_sigmoid_scalar((t - tau_off) / width)
            z = z + jump * acute_dir
            z = z + 0.08 * np.tanh(acute_mix @ z)
        # Subgroup 2: non-monotone pulse with partial recovery.
        else:
            rise = _stable_sigmoid_scalar((t - tau_on) / width)
            fall = _stable_sigmoid_scalar((t - tau_off) / width)
            pulse = amp * (rise - fall)
            z = z + pulse * acute_dir
            z = z + 0.05 * np.tanh(acute_mix @ z)

        return z

    if variant in ("regime_switch", "combo"):
        if "tau" not in patient_cache:
            patient_cache["tau"] = float(rng.uniform(0.20, 0.75) * patient_cache["t_event_ref"])
            patient_cache["rates2"] = rates * rng.uniform(0.5, 1.8, size=rates.shape[0])
            patient_cache["shift"] = rng.normal(0.0, 0.35, size=rates.shape[0])
            patient_cache["mix"] = rng.normal(0.0, 0.20, size=(rates.shape[0], rates.shape[0]))

        tau = patient_cache["tau"]
        if t > tau:
            z_tau = z0 * np.exp(-rates * tau)
            z = (z_tau + patient_cache["shift"]) * np.exp(-patient_cache["rates2"] * (t - tau))
        z = z + 0.12 * np.tanh(patient_cache["mix"] @ z)

    return z


def _sample_visit_times(rng, n_vis, t_event, variant):
    t_upper = max(1.0, float(t_event) * 0.85)

    if variant in ("rich_visits", "combo"):
        # More irregular than uniform: right-skewed over [0, t_upper], then sorted.
        u = rng.beta(0.9, 1.8, size=n_vis)
        t = np.sort(np.clip(u * t_upper + rng.normal(0.0, 0.01 * t_upper, size=n_vis), 0.0, t_upper))
    else:
        t = np.sort(rng.uniform(0.0, t_upper, size=n_vis))

    return t


def _summarize_latent_diagnostics(latent_diag_rows: list[dict], variant: str) -> dict:
    if not latent_diag_rows:
        return {
            "variant": variant,
            "n_patients": 0,
            "latent_range_l2_mean": float("nan"),
            "latent_range_l2_sd": float("nan"),
            "latent_range_l2_q25": float("nan"),
            "latent_range_l2_q50": float("nan"),
            "latent_range_l2_q75": float("nan"),
            "latent_max_step_mean": float("nan"),
            "latent_nonmonotone_ratio_mean": float("nan"),
            "group_counts": {},
            "group_fractions": {},
        }

    dfd = pd.DataFrame(latent_diag_rows)
    l2 = pd.to_numeric(dfd["latent_range_l2"], errors="coerce")
    max_step = pd.to_numeric(dfd["latent_max_step"], errors="coerce")
    nonmono = pd.to_numeric(dfd["latent_nonmonotone_ratio"], errors="coerce")
    group_counts = dfd["latent_group"].value_counts(dropna=False).to_dict() if "latent_group" in dfd.columns else {}
    n = max(int(len(dfd)), 1)
    group_fracs = {str(k): float(v) / float(n) for k, v in group_counts.items()}

    summary = {
        "variant": variant,
        "n_patients": int(len(dfd)),
        "latent_range_l2_mean": float(np.nanmean(l2)),
        "latent_range_l2_sd": float(np.nanstd(l2, ddof=1)) if int(np.sum(np.isfinite(l2))) > 1 else 0.0,
        "latent_range_l2_q25": float(np.nanquantile(l2, 0.25)),
        "latent_range_l2_q50": float(np.nanquantile(l2, 0.50)),
        "latent_range_l2_q75": float(np.nanquantile(l2, 0.75)),
        "latent_max_step_mean": float(np.nanmean(max_step)),
        "latent_nonmonotone_ratio_mean": float(np.nanmean(nonmono)),
        "group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "group_fractions": group_fracs,
    }
    return summary


def simulate_dataset(
    cfg: GenCfg,
    seed: int,
    scenario_name: str,
    mnar_strength: float,
    mar_strength: float,
    variant: str,
    return_diagnostics: bool = False,
):
    rng = np.random.default_rng(seed)

    cont_names = [f"X{i}" for i in range(cfg.n_cont)]
    cat_names = [f"CAT{i}" for i in range(cfg.n_cat)]

    # Global parameters.
    rates = rng.uniform(0.00005, 0.00025, size=cfg.d_latent)
    Wc = rng.normal(0.0, 1.0, size=(cfg.n_cont, cfg.d_latent))
    Wn = rng.normal(0.0, 0.8, size=(cfg.n_cont, cfg.d_latent))
    vb = rng.normal(0.0, 1.0, size=(cfg.n_cat, cfg.d_latent))
    beta = rng.normal(0.0, 0.7, size=cfg.d_latent)
    u = rng.normal(0.0, 1.0, size=cfg.d_latent)
    u = u / (np.linalg.norm(u) + 1e-12)

    # Missingness params (continuous).
    a = rng.normal(-2.2, 0.3, size=cfg.n_cont)
    b = rng.normal(0.7, 0.2, size=cfg.n_cont)
    c = rng.normal(1.0, 0.2, size=cfg.n_cont)

    mar_a = rng.normal(-2.0, 0.3, size=cfg.n_cont)
    mar_b = rng.normal(0.5, 0.2, size=cfg.n_cont)

    # Optional categorical-missingness params.
    cat_miss_a = rng.normal(-3.0, 0.25, size=cfg.n_cat)
    cat_miss_b = rng.normal(0.45, 0.15, size=cfg.n_cat)

    # Hazard integration grid.
    t_grid = np.arange(0, cfg.t_max_days + 1, cfg.dt_days)

    rows = []
    latent_diag_rows = []
    is_mnar_corr = bool(scenario_name == "scenario_MNAR_CORRELATED")
    mnar_corr_chol = None
    if is_mnar_corr and cfg.mnar_corr_state_dependent == 0:
        mnar_corr_chol = _build_corr_chol(
            n_features=cfg.n_cont,
            rho=cfg.mnar_corr_rho,
            structure=cfg.mnar_corr_structure,
            block_size=cfg.mnar_corr_block_size,
        )

    for pid in range(cfg.n_patients):
        z0 = rng.normal(0.0, 1.0, size=cfg.d_latent)

        if variant in ("rich_visits", "combo"):
            n_vis = int(rng.integers(max(3, cfg.min_visits), max(cfg.max_visits, 10) + 1))
        else:
            n_vis = int(rng.integers(cfg.min_visits, cfg.max_visits + 1))

        # Event time from hazard over latent trajectory.
        z_grid = np.zeros((len(t_grid), cfg.d_latent), dtype=float)
        cache = {"t_event_ref": float(cfg.t_max_days)}
        for i, tg in enumerate(t_grid):
            z_grid[i] = _latent_at_t(float(tg), z0, rates, variant, rng, cache)

        # Latent dynamic-range diagnostics on generator time grid.
        z_min = np.nanmin(z_grid, axis=0)
        z_max = np.nanmax(z_grid, axis=0)
        z_span = z_max - z_min
        z_diff = np.diff(z_grid, axis=0)
        if z_diff.shape[0] >= 2:
            # Fraction of adjacent derivative sign changes, averaged over latent dimensions.
            sign_change_num = np.sum((z_diff[1:, :] * z_diff[:-1, :]) < 0.0, axis=0)
            sign_change_den = np.maximum(z_diff.shape[0] - 1, 1)
            nonmono_ratio = float(np.mean(sign_change_num / sign_change_den))
        else:
            nonmono_ratio = 0.0

        latent_diag_rows.append(
            {
                "RANDID": int(pid),
                "latent_group": str(cache.get("acute_group_name", "base_like")),
                "latent_range_l2": float(np.linalg.norm(z_span)),
                "latent_range_mean_abs": float(np.mean(np.abs(z_span))),
                "latent_max_step": float(np.max(np.linalg.norm(z_diff, axis=1))) if z_diff.size else 0.0,
                "latent_nonmonotone_ratio": nonmono_ratio,
            }
        )

        linpred = z_grid @ beta
        haz = cfg.lam0 * np.exp(linpred)
        cum_haz = np.cumsum(haz) * cfg.dt_days
        thr = -np.log(1.0 - rng.uniform(0.0, 1.0))
        idx = int(np.searchsorted(cum_haz, thr))
        # True event time before censoring/admin truncation.
        if idx >= len(t_grid):
            t_event_true = float(cfg.t_max_days + cfg.dt_days)
        else:
            t_event_true = float(t_grid[idx])

        # Default (administrative) censoring at t_max.
        t_censor = float(cfg.t_max_days)
        if variant == "informative_censor":
            # Lightweight informative-censoring mechanism:
            # higher latent risk -> higher dropout hazard -> earlier censoring.
            risk_c = float(u @ z0)
            lam_c = float(cfg.censor_lam0 * np.exp(cfg.censor_risk_gamma * risk_c))
            lam_c = float(np.clip(lam_c, 1e-8, 5e-3))
            t_censor = float(rng.exponential(scale=1.0 / lam_c))
            t_censor = float(np.clip(t_censor, cfg.censor_min_days, cfg.t_max_days))

        t_obs = float(min(t_event_true, t_censor, cfg.t_max_days))
        death = int((t_event_true <= t_censor) and (t_event_true <= cfg.t_max_days))

        cache["t_event_ref"] = t_obs
        cache["t_event_true"] = t_event_true
        cache["t_censor"] = t_censor
        cache["informative_censor"] = int(variant == "informative_censor")

        t_vis = _sample_visit_times(rng, n_vis, t_obs, variant)

        for t in t_vis:
            if t >= t_obs:
                break

            time_bin = int(np.floor(np.round(t) / 10.0))
            timedth_bin = int(np.ceil(np.round(t_obs) / 10.0))
            if timedth_bin - time_bin <= cfg.min_gap_bins:
                continue

            zt = _latent_at_t(float(t), z0, rates, variant, rng, cache)

            if variant in ("nonlinear_obs", "combo"):
                x_mean = (Wc @ zt) + (0.35 * cfg.nonlinear_alpha) * np.tanh(Wn @ zt)
            else:
                x_mean = Wc @ zt

            if variant in ("heavy_tail", "combo"):
                eps = 0.30 * rng.standard_t(df=3, size=cfg.n_cont)
                outlier = rng.uniform(0.0, 1.0, size=cfg.n_cont) < 0.01
                eps[outlier] += rng.normal(0.0, 4.0, size=int(outlier.sum()))
                x_cont = x_mean + eps
            elif variant == "nonlinear_obs":
                # `nonlinear_alpha` interpolates between base-like Gaussian observations and nonlinear heavy-tailed observations.
                alpha_nl = float(np.clip(cfg.nonlinear_alpha, 0.0, 1.0))
                if alpha_nl <= 0.0:
                    x_cont = x_mean + rng.normal(0.0, 0.5, size=cfg.n_cont)
                else:
                    eps = (0.30 * alpha_nl) * rng.standard_t(df=3, size=cfg.n_cont)
                    outlier = rng.uniform(0.0, 1.0, size=cfg.n_cont) < (0.01 * alpha_nl)
                    if outlier.any():
                        eps[outlier] += rng.normal(0.0, 4.0 * alpha_nl, size=int(outlier.sum()))
                    # Residual Gaussian term keeps a smooth interpolation to base.
                    x_cont = x_mean + eps + rng.normal(0.0, 0.5 * (1.0 - alpha_nl), size=cfg.n_cont)
            else:
                x_cont = x_mean + rng.normal(0.0, 0.5, size=cfg.n_cont)

            p_cat = sigmoid(vb @ zt)
            x_cat = (rng.uniform(0.0, 1.0, size=cfg.n_cat) < p_cat).astype(int)

            # Continuous missingness mechanisms.
            miss_cont = np.zeros(cfg.n_cont, dtype=bool)
            if mnar_strength > 0:
                severity = float(u @ zt)
                near = 0.0
                if death == 1:
                    near = max(0.0, 1.0 - (t_obs - t) / cfg.near_event_window_days)
                logits = a + mnar_strength * (b * severity + c * near)
                p_miss = sigmoid(logits)

                # Independent MNAR sampling:
                # miss_cont = rng.uniform(0.0, 1.0, size=cfg.n_cont) < p_miss
                if is_mnar_corr:
                    if cfg.mnar_corr_state_dependent == 1:
                        rho_t = float(
                            np.clip(
                                cfg.mnar_corr_rho + cfg.mnar_corr_state_scale * np.tanh(severity),
                                0.0,
                                0.95,
                            )
                        )
                        chol_t = _build_corr_chol(
                            n_features=cfg.n_cont,
                            rho=rho_t,
                            structure=cfg.mnar_corr_structure,
                            block_size=cfg.mnar_corr_block_size,
                        )
                        miss_cont = _sample_correlated_bernoulli_gaussian_copula(
                            rng=rng,
                            p_miss=p_miss,
                            chol=chol_t,
                        )
                    else:
                        miss_cont = _sample_correlated_bernoulli_gaussian_copula(
                            rng=rng,
                            p_miss=p_miss,
                            chol=mnar_corr_chol,
                        )
                else:
                    miss_cont = rng.uniform(0.0, 1.0, size=cfg.n_cont) < p_miss

            if mar_strength > 0:
                cat_signal = float(x_cat.mean())
                logits_mar = mar_a + mar_strength * mar_b * cat_signal
                p_miss_mar = sigmoid(logits_mar)
                miss_cont = miss_cont | (rng.uniform(0.0, 1.0, size=cfg.n_cont) < p_miss_mar)

            # Optional categorical missingness for sensitivity analyses.
            miss_cat = np.zeros(cfg.n_cat, dtype=bool)
            if cfg.mask_categorical == 1 and (mnar_strength > 0 or mar_strength > 0):
                logits_cat = cat_miss_a + (mnar_strength + mar_strength) * cat_miss_b * float(np.abs(zt).mean())
                p_cat_miss = sigmoid(logits_cat)
                miss_cat = rng.uniform(0.0, 1.0, size=cfg.n_cat) < p_cat_miss

            out = {
                "RANDID": int(pid),
                "TIME": int(np.round(t)),
                "TIMEDTH": int(np.round(t_obs)),
                "DEATH": int(death),
            }
            for j, name in enumerate(cont_names):
                out[name] = np.nan if miss_cont[j] else float(x_cont[j])
            for j, name in enumerate(cat_names):
                if miss_cat[j]:
                    out[name] = np.nan
                else:
                    out[name] = int(x_cat[j])

            rows.append(out)

    df = pd.DataFrame(rows)
    df = df.sort_values(["RANDID", "TIME"]).reset_index(drop=True)
    latent_summary = _summarize_latent_diagnostics(latent_diag_rows, variant=variant)
    if return_diagnostics:
        return df, latent_summary

    return df


def split_ids(ids: np.ndarray, test_frac: float, train_frac: float, split_seed: int, cohort_seed: int):
    rng = np.random.default_rng(split_seed + cohort_seed)
    ids = np.array(sorted(ids))
    n_test = int(len(ids) * test_frac)
    test_ids = set(rng.choice(ids, size=n_test, replace=False))
    rem = np.array(sorted(set(ids) - test_ids))
    n_train = int(len(rem) * train_frac / (1.0 - test_frac))
    train_ids = set(rng.choice(rem, size=n_train, replace=False))
    val_ids = set(ids) - test_ids - train_ids
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def save_bundle(
    df,
    out_dir,
    cfg: GenCfg,
    scenario_name: str,
    variant: str,
    cohort_seed: int,
    gen_seed: int,
    mnar_strength: float,
    mar_strength: float,
    latent_dynamics_summary: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    train_ids, val_ids, test_ids = split_ids(
        ids=df["RANDID"].unique(),
        test_frac=cfg.test_frac,
        train_frac=cfg.train_frac,
        split_seed=cfg.split_seed,
        cohort_seed=cohort_seed,
    )

    df_train = df[df["RANDID"].isin(train_ids)].copy()
    df_val = df[df["RANDID"].isin(val_ids)].copy()
    df_test = df[df["RANDID"].isin(test_ids)].copy()

    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    gen_config = asdict(cfg)
    corr_blocks = _corr_blocks_from_size(cfg.n_cont, cfg.mnar_corr_block_size)
    gen_config.update(
        {
            "scenario": scenario_name,
            "variant": variant,
            "cohort_seed": int(cohort_seed),
            "gen_seed": int(gen_seed),
            "mnar_strength": float(mnar_strength),
            "mar_strength": float(mar_strength),
            "feat_cont": [f"X{i}" for i in range(cfg.n_cont)],
            "feat_cat": [f"CAT{i}" for i in range(cfg.n_cat)],
            "mask_pool": [f"X{i}" for i in range(min(cfg.mask_pool_size, cfg.n_cont))],
            "id_col": "RANDID",
            "time_col": "TIME",
            "ttilde_col": "TIMEDTH",
            "event_col": "DEATH",
            "censor_lam0": float(cfg.censor_lam0),
            "censor_risk_gamma": float(cfg.censor_risk_gamma),
            "censor_min_days": int(cfg.censor_min_days),
            "mnar_corr_enabled": int(scenario_name == "scenario_MNAR_CORRELATED"),
            "mnar_corr_rho": float(cfg.mnar_corr_rho),
            "mnar_corr_structure": str(cfg.mnar_corr_structure),
            "mnar_corr_block_size": int(cfg.mnar_corr_block_size),
            "mnar_corr_blocks": corr_blocks if str(cfg.mnar_corr_structure) == "block" else [],
            "mnar_corr_state_dependent": int(cfg.mnar_corr_state_dependent),
            "mnar_corr_state_scale": float(cfg.mnar_corr_state_scale),
            "nonlinear_alpha": float(cfg.nonlinear_alpha),
        }
    )
    with open(os.path.join(out_dir, "gen_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    meta = {
        "scenario": scenario_name,
        "variant": variant,
        "seed": int(gen_seed),
        "N": int(df["RANDID"].nunique()),
        "event_rate": float(df.groupby("RANDID")["DEATH"].first().mean()),
        "feat_cont": [f"X{i}" for i in range(cfg.n_cont)],
        "feat_cat": [f"CAT{i}" for i in range(cfg.n_cat)],
        "mask_pool": [f"X{i}" for i in range(min(cfg.mask_pool_size, cfg.n_cont))],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cohort_stats = {
        "n_train": int(len(train_ids)),
        "n_valid": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "event_rate": float(df.groupby("RANDID")["DEATH"].first().mean()),
        "rows_train": int(len(df_train)),
        "rows_valid": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "visits_per_patient_train_median": float(df_train.groupby("RANDID")["TIME"].count().median()),
        "visits_per_patient_train_iqr": [
            float(df_train.groupby("RANDID")["TIME"].count().quantile(0.25)),
            float(df_train.groupby("RANDID")["TIME"].count().quantile(0.75)),
        ],
        "censor_rate": float(1.0 - df.groupby("RANDID")["DEATH"].first().mean()),
    }
    with open(os.path.join(out_dir, "cohort_stats.json"), "w") as f:
        json.dump(cohort_stats, f, indent=2)

    if latent_dynamics_summary is not None:
        # Optional diagnostic artifact; split files keep the public release schema.
        with open(os.path.join(out_dir, "latent_dynamics_summary.json"), "w") as f:
            json.dump(latent_dynamics_summary, f, indent=2)


def _cohort_dir(out_root: str, variant: str, scenario_name: str, cohort_seed: int) -> str:
    return os.path.join(out_root, variant, scenario_name, f"cohortseed_{cohort_seed:03d}")


def _require_split_files(cohort_dir: str) -> None:
    needed = ["train.csv", "val.csv", "test.csv", "gen_config.json"]
    missing = [x for x in needed if not os.path.exists(os.path.join(cohort_dir, x))]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under {cohort_dir}")


def _safe_read_csv(fp: str) -> pd.DataFrame:
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def save_transfer_bundle(
    out_root: str,
    variant: str,
    scenario_name: str,
    cohort_seed: int,
    train_scenario: str,
    test_scenario: str,
):
    """
    Build a transfer scenario by composing:
      - train/val from `train_scenario`
      - test from `test_scenario`
    This creates a train-test source change without touching the evaluation protocol.
    """
    train_dir = _cohort_dir(out_root, variant, train_scenario, cohort_seed)
    test_dir = _cohort_dir(out_root, variant, test_scenario, cohort_seed)
    out_dir = _cohort_dir(out_root, variant, scenario_name, cohort_seed)
    os.makedirs(out_dir, exist_ok=True)

    _require_split_files(train_dir)
    _require_split_files(test_dir)

    df_train = _safe_read_csv(os.path.join(train_dir, "train.csv"))
    df_val = _safe_read_csv(os.path.join(train_dir, "val.csv"))
    df_test = _safe_read_csv(os.path.join(test_dir, "test.csv"))

    # Offset test IDs to avoid accidental overlap with train/val patient IDs.
    if "RANDID" in df_test.columns and not df_test.empty:
        max_tid = int(max(df_train["RANDID"].max(), df_val["RANDID"].max()))
        id_offset = max(max_tid + 1, 10_000_000)
        df_test["RANDID"] = df_test["RANDID"].astype(np.int64) + int(id_offset)

    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    with open(os.path.join(train_dir, "gen_config.json"), "r") as f:
        cfg_train = json.load(f)
    with open(os.path.join(test_dir, "gen_config.json"), "r") as f:
        cfg_test = json.load(f)

    gen_config = dict(cfg_train)
    gen_config.update(
        {
            "scenario": scenario_name,
            "variant": variant,
            "cohort_seed": int(cohort_seed),
            "transfer_scenario": 1,
            "transfer_train_scenario": train_scenario,
            "transfer_test_scenario": test_scenario,
            "transfer_note": "train+val copied from train_scenario; test copied from test_scenario",
            "train_gen_seed": cfg_train.get("gen_seed"),
            "test_gen_seed": cfg_test.get("gen_seed"),
            "train_mnar_strength": cfg_train.get("mnar_strength"),
            "train_mar_strength": cfg_train.get("mar_strength"),
            "test_mnar_strength": cfg_test.get("mnar_strength"),
            "test_mar_strength": cfg_test.get("mar_strength"),
            "mnar_strength": cfg_test.get("mnar_strength", cfg_train.get("mnar_strength", 0.0)),
            "mar_strength": cfg_test.get("mar_strength", cfg_train.get("mar_strength", 0.0)),
        }
    )
    with open(os.path.join(out_dir, "gen_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    event_rate = float(df_all.groupby("RANDID")["DEATH"].first().mean()) if not df_all.empty else float("nan")
    meta = {
        "scenario": scenario_name,
        "variant": variant,
        "cohort_seed": int(cohort_seed),
        "N": int(df_all["RANDID"].nunique()) if "RANDID" in df_all.columns else 0,
        "event_rate": event_rate,
        "transfer_scenario": 1,
        "transfer_train_scenario": train_scenario,
        "transfer_test_scenario": test_scenario,
        "feat_cont": gen_config.get("feat_cont", [f"X{i}" for i in range(int(gen_config.get("n_cont", 12)))]),
        "feat_cat": gen_config.get("feat_cat", [f"CAT{i}" for i in range(int(gen_config.get("n_cat", 4)))]),
        "mask_pool": gen_config.get("mask_pool", [f"X{i}" for i in range(min(int(gen_config.get("mask_pool_size", 6)), int(gen_config.get("n_cont", 12))))]),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cohort_stats = {
        "n_train": int(df_train["RANDID"].nunique()) if "RANDID" in df_train.columns else 0,
        "n_valid": int(df_val["RANDID"].nunique()) if "RANDID" in df_val.columns else 0,
        "n_test": int(df_test["RANDID"].nunique()) if "RANDID" in df_test.columns else 0,
        "event_rate": event_rate,
        "rows_train": int(len(df_train)),
        "rows_valid": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "visits_per_patient_train_median": float(df_train.groupby("RANDID")["TIME"].count().median()) if not df_train.empty else float("nan"),
        "visits_per_patient_train_iqr": [
            float(df_train.groupby("RANDID")["TIME"].count().quantile(0.25)) if not df_train.empty else float("nan"),
            float(df_train.groupby("RANDID")["TIME"].count().quantile(0.75)) if not df_train.empty else float("nan"),
        ],
        "censor_rate": float(1.0 - event_rate) if np.isfinite(event_rate) else float("nan"),
        "transfer_train_source": train_scenario,
        "transfer_test_source": test_scenario,
    }
    with open(os.path.join(out_dir, "cohort_stats.json"), "w") as f:
        json.dump(cohort_stats, f, indent=2)


def scenario_strengths(scenario_name: str):
    table = {
        "scenario_A": (0.0, 0.0),
        "scenario_B": (0.8, 0.0),
        "scenario_C": (1.6, 0.0),
        "scenario_MAR": (0.0, 0.8),
        "scenario_VISITSHIFT": (0.0, 0.0),
        "scenario_MISMATCH": (0.8, 0.0),
        # Matched to MNAR Mild for clean paired comparison (marginals held by copula).
        "scenario_MNAR_CORRELATED": (0.8, 0.0),
        # Transfer scenarios are composed from two existing bundles (no direct simulation here).
        "scenario_VISITSHIFT_TRANSFER": (0.0, 0.0),
        "scenario_MISMATCH_TRANSFER": (0.8, 0.0),
    }
    if scenario_name in MNAR_ALPHA_SCENARIO_MAP:
        return (float(MNAR_ALPHA_SCENARIO_MAP[scenario_name]), 0.0)
    return table.get(scenario_name, (0.0, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "Data", "SynLV_Zoo"))
    ap.add_argument(
        "--variant",
        type=str,
        default="base",
        # Primary hosted scenarios use `variant="base"`. Other variants are
        # source-reproducible sensitivity branches used for appendix analyses.
        # `high_visits` keeps base dynamics/observation model; only visit-count range changes.
        choices=[
            "base",
            "high_visits",
            "regime_switch",
            "acute_hetero",
            # Block-correlated MNAR variant with the base latent and observation generator.
            "mnar_corr_block",
            "nonlinear_obs",
            "informative_censor",
            "heavy_tail",
            "rich_visits",
            "combo",
            "dz8",
        ],
    )
    ap.add_argument(
        "--scenarios",
        type=str,
        default="scenario_A,scenario_B,scenario_C,scenario_MNAR_CORRELATED,scenario_MAR,scenario_VISITSHIFT,scenario_VISITSHIFT_TRANSFER,scenario_MISMATCH,scenario_MISMATCH_TRANSFER",
    )
    ap.add_argument("--cohort_seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--n_patients", type=int, default=8000)
    ap.add_argument("--d_latent", type=int, default=4)
    ap.add_argument("--mask_categorical", type=int, default=0)
    ap.add_argument("--min_visits", type=int, default=3)
    ap.add_argument("--max_visits", type=int, default=6)
    ap.add_argument("--censor_lam0", type=float, default=5.0e-5)
    ap.add_argument("--censor_risk_gamma", type=float, default=0.9)
    ap.add_argument("--censor_min_days", type=int, default=120)
    ap.add_argument("--mnar_corr_rho", type=float, default=0.55)
    ap.add_argument("--mnar_corr_structure", type=str, default="exchangeable", choices=["exchangeable", "block"])
    ap.add_argument("--mnar_corr_block_size", type=int, default=6)
    ap.add_argument("--mnar_corr_state_dependent", type=int, default=0)
    ap.add_argument("--mnar_corr_state_scale", type=float, default=0.10)
    ap.add_argument("--nonlinear_alpha", type=float, default=1.0)
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    cohort_seeds = [int(x.strip()) for x in args.cohort_seeds.split(",") if x.strip()]

    d_latent = int(args.d_latent)
    if args.variant == "dz8":
        d_latent = 8

    cfg = GenCfg(
        n_patients=int(args.n_patients),
        d_latent=d_latent,
        min_visits=int(args.min_visits),
        max_visits=int(args.max_visits),
        mask_categorical=int(args.mask_categorical),
        censor_lam0=float(args.censor_lam0),
        censor_risk_gamma=float(args.censor_risk_gamma),
        censor_min_days=int(args.censor_min_days),
        mnar_corr_rho=float(args.mnar_corr_rho),
        mnar_corr_structure=str(args.mnar_corr_structure),
        mnar_corr_block_size=int(args.mnar_corr_block_size),
        mnar_corr_state_dependent=int(args.mnar_corr_state_dependent),
        mnar_corr_state_scale=float(args.mnar_corr_state_scale),
        nonlinear_alpha=float(args.nonlinear_alpha),
    )

    base_seed_map = {
        "scenario_A": 1,
        "scenario_B": 2,
        "scenario_C": 3,
        "scenario_MAR": 4,
        "scenario_VISITSHIFT": 5,
        "scenario_MISMATCH": 6,
        "scenario_MNAR_CORRELATED": 7,
        "scenario_VISITSHIFT_TRANSFER": 8,
        "scenario_MISMATCH_TRANSFER": 9,
    }
    # Deterministic seed ids for MNAR-alpha ladder.
    for i, sname in enumerate(sorted(MNAR_ALPHA_SCENARIO_MAP.keys()), start=10):
        base_seed_map[sname] = i

    print("=" * 70)
    print("SynLV generator zoo")
    print(f"variant={args.variant}")
    print(f"out_root={args.out_root}")
    print("=" * 70)

    for scenario_name in scenarios:
        print("\n" + "-" * 70)

        # Transfer scenarios: compose train/val from base scenario and test from shifted scenario.
        if scenario_name in TRANSFER_SCENARIO_MAP:
            train_src, test_src = TRANSFER_SCENARIO_MAP[scenario_name]
            print(f"scenario={scenario_name} (transfer: train={train_src}, test={test_src})")

            for cs in cohort_seeds:
                # Ensure both source bundles exist (generate on-demand if missing).
                for src_scenario in (train_src, test_src):
                    src_dir = _cohort_dir(args.out_root, args.variant, src_scenario, int(cs))
                    src_train_fp = os.path.join(src_dir, "train.csv")
                    src_val_fp = os.path.join(src_dir, "val.csv")
                    src_test_fp = os.path.join(src_dir, "test.csv")
                    src_cfg_fp = os.path.join(src_dir, "gen_config.json")
                    if not (
                        os.path.exists(src_train_fp)
                        and os.path.exists(src_val_fp)
                        and os.path.exists(src_test_fp)
                        and os.path.exists(src_cfg_fp)
                    ):
                        src_mnar, src_mar = scenario_strengths(src_scenario)
                        src_seed = base_seed_map.get(src_scenario, 42) * 1000 + int(cs)
                        print(f"  cohort_seed={cs} source_missing={src_scenario} -> generating seed={src_seed}")
                        df_src, latent_diag_src = simulate_dataset(
                            cfg=cfg,
                            seed=src_seed,
                            scenario_name=src_scenario,
                            mnar_strength=src_mnar,
                            mar_strength=src_mar,
                            variant=args.variant,
                            return_diagnostics=True,
                        )
                        save_bundle(
                            df=df_src,
                            out_dir=src_dir,
                            cfg=cfg,
                            scenario_name=src_scenario,
                            variant=args.variant,
                            cohort_seed=int(cs),
                            gen_seed=int(src_seed),
                            mnar_strength=src_mnar,
                            mar_strength=src_mar,
                            latent_dynamics_summary=latent_diag_src,
                        )
                        print(f"    source saved rows={len(df_src)} patients={df_src['RANDID'].nunique()} -> {src_dir}")

                save_transfer_bundle(
                    out_root=args.out_root,
                    variant=args.variant,
                    scenario_name=scenario_name,
                    cohort_seed=int(cs),
                    train_scenario=train_src,
                    test_scenario=test_src,
                )
                out_dir = _cohort_dir(args.out_root, args.variant, scenario_name, int(cs))
                print(f"  cohort_seed={cs} transfer bundle saved -> {out_dir}")
            continue

        mnar_strength, mar_strength = scenario_strengths(scenario_name)
        base_seed = base_seed_map.get(scenario_name, 42)
        print(f"scenario={scenario_name} mnar={mnar_strength} mar={mar_strength}")
        for cs in cohort_seeds:
            gen_seed = base_seed * 1000 + int(cs)
            print(f"  cohort_seed={cs} gen_seed={gen_seed}")

            df, latent_diag = simulate_dataset(
                cfg=cfg,
                seed=gen_seed,
                scenario_name=scenario_name,
                mnar_strength=mnar_strength,
                mar_strength=mar_strength,
                variant=args.variant,
                return_diagnostics=True,
            )

            out_dir = os.path.join(args.out_root, args.variant, scenario_name, f"cohortseed_{cs:03d}")
            save_bundle(
                df=df,
                out_dir=out_dir,
                cfg=cfg,
                scenario_name=scenario_name,
                variant=args.variant,
                cohort_seed=int(cs),
                gen_seed=int(gen_seed),
                mnar_strength=mnar_strength,
                mar_strength=mar_strength,
                latent_dynamics_summary=latent_diag,
            )
            print(f"    saved rows={len(df)} patients={df['RANDID'].nunique()} -> {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
