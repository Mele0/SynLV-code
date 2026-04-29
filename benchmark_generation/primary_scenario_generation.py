"""Primary source-cohort generator for SynLV v1.0.

This module provides the release-compatible simulator used by
`benchmark_generation/final_generation.py` for primary source cohorts stored
under local generation roots such as `Data/SynLV/<scenario>/cohortseed_NNN/`.
"""
import os, json
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT = os.environ.get("SYNLV_FINALGEN_OUT_ROOT", os.path.join(REPO_ROOT, "Data", "SynLV"))

N_PATIENTS    = 8000
N_CONT        = 12     # dc: number of continuous features (X0-X11)
N_CAT         = 4      # dk: number of categorical features (CAT0-CAT3)
MASK_POOL     = 6      # first 6 continuous features are maskable
T_MAX_DAYS    = 9000
MIN_VISITS    = 3
MAX_VISITS    = 6
LAM0          = 1e-4
DT_DAYS       = 10
D_LATENT      = 4
NEAR_EVENT_WINDOW = 600
MIN_GAP_BINS  = 3

# Scenarios: name -> (seed, default mnar_strength)
SCENARIOS = {
    "scenario_A":          (1, 0.0),
    "scenario_B":          (2, 0.8),
    "scenario_C":          (3, 1.6),
    "scenario_MAR":        (4, 0.0),   # MAR handled separately below
    "scenario_VISITSHIFT": (5, 0.0),
    "scenario_MISMATCH":   (6, 0.0),
}

COHORT_SEEDS = [0, 1, 2, 3, 4]

TEST_FRAC  = 0.20
TRN_FRAC = 0.65
SPLIT_SEED = 1991


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate_dataset(
    n_patients=N_PATIENTS,
    n_cont=N_CONT,
    n_cat=N_CAT,
    d_latent=D_LATENT,
    t_max_days=T_MAX_DAYS,
    min_visits=MIN_VISITS,
    max_visits=MAX_VISITS,
    seed=0,
    lam0=LAM0,
    dt_days=DT_DAYS,
    mnar_strength=0.0,
    mar_strength=0.0,
    near_event_window_days=NEAR_EVENT_WINDOW,
    min_gap_bins=MIN_GAP_BINS,
):
    rng = np.random.default_rng(seed)

    cont_names = [f"X{i}"   for i in range(n_cont)]
    cat_names  = [f"CAT{i}" for i in range(n_cat)]

    # Global parameters
    rates = rng.uniform(0.00005, 0.00025, size=d_latent)
    Wc    = rng.normal(0, 1.0, size=(n_cont, d_latent))
    vb    = rng.normal(0, 1.0, size=(n_cat,  d_latent))
    beta  = rng.normal(0, 0.7, size=d_latent)
    u     = rng.normal(0, 1.0, size=d_latent)
    u     = u / (np.linalg.norm(u) + 1e-12)

    # MNAR params
    a = rng.normal(-2.2, 0.3, size=n_cont)
    b = rng.normal( 0.7, 0.2, size=n_cont)
    c = rng.normal( 1.0, 0.2, size=n_cont)

    # MAR params
    mar_a = rng.normal(-2.0, 0.3, size=n_cont)
    mar_b = rng.normal( 0.5, 0.2, size=n_cont)

    rows   = []
    t_grid = np.arange(0, t_max_days + 1, dt_days)

    for pid in range(n_patients):
        z0    = rng.normal(0, 1.0, size=d_latent)
        n_vis = int(rng.integers(min_visits, max_visits + 1))

        # Compute event time
        z_grid  = z0 * np.exp(-rates * t_grid[:, None])
        linpred = z_grid @ beta
        haz     = lam0 * np.exp(linpred)
        cum_haz = np.cumsum(haz) * dt_days
        thr     = -np.log(1.0 - rng.uniform(0, 1))
        idx     = np.searchsorted(cum_haz, thr)
        if idx >= len(t_grid):
            T_event = float(t_max_days)
            event   = 0
        else:
            T_event = float(t_grid[idx])
            event   = 1

        T_obs = T_event
        DEATH = event

        # Sample visits within 85% of event time
        t_upper = max(T_event * 0.85, 1.0)
        t_vis   = np.sort(rng.uniform(0, t_upper, size=n_vis))

        for t in t_vis:
            if t >= T_event:
                break

            time_bin    = int(np.floor(np.round(t)     / 10))
            timedth_bin = int(np.ceil( np.round(T_obs) / 10))

            if timedth_bin - time_bin <= min_gap_bins:
                continue

            zt     = z0 * np.exp(-rates * t)
            x_cont = (Wc @ zt) + rng.normal(0, 0.5, size=n_cont)
            p_cat  = sigmoid(vb @ zt)
            x_cat  = (rng.uniform(0, 1, size=n_cat) < p_cat).astype(int)

            # MNAR missingness
            miss = np.zeros(n_cont, dtype=bool)
            if mnar_strength > 0:
                severity = float(u @ zt)
                near = 0.0
                if event == 1:
                    near = max(0.0, 1.0 - (T_event - t) / near_event_window_days)
                logits = a + mnar_strength * (b * severity + c * near)
                p_miss = sigmoid(logits)
                miss   = rng.uniform(0, 1, size=n_cont) < p_miss

            # MAR missingness (depends on observed cat features only)
            if mar_strength > 0:
                cat_signal = x_cat.astype(float).mean()
                logits_mar = mar_a + mar_strength * mar_b * cat_signal
                p_miss_mar = sigmoid(logits_mar)
                miss_mar   = rng.uniform(0, 1, size=n_cont) < p_miss_mar
                miss       = miss | miss_mar

            out = {
                "RANDID":  pid,
                "TIME":    int(np.round(t)),
                "TIMEDTH": int(np.round(T_obs)),
                "DEATH":   int(DEATH),
            }
            for j, name in enumerate(cont_names):
                out[name] = np.nan if miss[j] else float(x_cont[j])
            for j, name in enumerate(cat_names):
                out[name] = int(x_cat[j])

            rows.append(out)

    df = pd.DataFrame(rows)
    df = df.sort_values(["RANDID", "TIME"]).reset_index(drop=True)
    return df


# SPLIT & SAVE
def split_and_save(df, out_dir, cohort_seed, scenario_name, mnar_strength, mar_strength, gen_seed):
    np.random.seed(SPLIT_SEED + cohort_seed)
    ids      = np.array(sorted(df["RANDID"].unique()))
    test_ids = set(np.random.choice(ids, size=int(len(ids) * TEST_FRAC), replace=False))
    rem_ids  = np.array(sorted(set(ids) - test_ids))
    train_size = int(len(rem_ids) * TRN_FRAC / (1 - TEST_FRAC))
    train_ids  = set(np.random.choice(rem_ids, size=train_size, replace=False))
    valid_ids  = set(ids) - test_ids - train_ids

    df_train = df[df["RANDID"].isin(train_ids)].copy()
    df_valid = df[df["RANDID"].isin(valid_ids)].copy()
    df_test  = df[df["RANDID"].isin(test_ids) ].copy()

    os.makedirs(out_dir, exist_ok=True)
    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_valid.to_csv(os.path.join(out_dir, "val.csv"),   index=False)
    df_test.to_csv( os.path.join(out_dir, "test.csv"),  index=False)

    # gen_config.json
    gen_config = {
        "scenario":       scenario_name,
        "cohort_seed":    cohort_seed,
        "gen_seed":       gen_seed,
        "N":              int(df["RANDID"].nunique()),
        "dz":             D_LATENT,
        "dc":             N_CONT,
        "dk":             N_CAT,
        "mask_pool_size": MASK_POOL,
        "t_max":          T_MAX_DAYS,
        "dt_days":        DT_DAYS,
        "visit_min":      MIN_VISITS,
        "visit_max":      MAX_VISITS,
        "lam0":           LAM0,
        "min_gap_bins":   MIN_GAP_BINS,
        "mnar_strength":  mnar_strength,
        "mar_strength":   mar_strength,
        "id_col":         "RANDID",
        "time_col":       "TIME",
        "ttilde_col":     "TIMEDTH",
        "event_col":      "DEATH",
    }
    with open(os.path.join(out_dir, "gen_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    # meta.json
    meta = {
        "scenario":   scenario_name,
        "seed":       gen_seed,
        "N":          int(df["RANDID"].nunique()),
        "dz":         D_LATENT,
        "dc":         N_CONT,
        "dk":         N_CAT,
        "t_max":      T_MAX_DAYS,
        "visit_min":  MIN_VISITS,
        "visit_max":  MAX_VISITS,
        "feat_cont":  [f"X{i}" for i in range(N_CONT)],
        "feat_cat":   [f"CAT{i}" for i in range(N_CAT)],
        "mask_pool":  [f"X{i}" for i in range(MASK_POOL)],
        "id_col":     "RANDID",
        "time_col":   "TIME",
        "ttilde_col": "TIMEDTH",
        "event_col":  "DEATH",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # cohort_stats.json
    last = df_train.groupby("RANDID")["TIMEDTH"].first()
    stats = {
        "n_train":       len(train_ids),
        "n_valid":       len(valid_ids),
        "n_test":        len(test_ids),
        "event_rate":    float(df.groupby("RANDID")["DEATH"].first().mean()),
        "timedth_mean":  float(last.mean()),
        "timedth_median":float(last.median()),
        "timedth_max":   float(last.max()),
    }
    with open(os.path.join(out_dir, "cohort_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"    Saved: train={len(df_train)}, val={len(df_valid)}, test={len(df_test)} rows")
    print(f"    Patients: train={len(train_ids)}, val={len(valid_ids)}, test={len(test_ids)}")
    print(f"    Event rate: {stats['event_rate']:.2f}")


def get_scenario_params(scenario_name):
    """Return (mnar_strength, mar_strength) for each scenario."""
    return {
        "scenario_A":          (0.0, 0.0),   # Reference
        "scenario_B":          (0.8, 0.0),   # MNAR mild
        "scenario_C":          (1.6, 0.0),   # MNAR strong
        "scenario_MAR":        (0.0, 0.8),   # MAR
        "scenario_VISITSHIFT": (0.0, 0.0),   # visit-schedule control under the base sampler
        "scenario_MISMATCH":   (0.8, 0.0),   # alternative MNAR observation regime
    }.get(scenario_name, (0.0, 0.0))


if __name__ == "__main__":
    print("SynLV primary source cohort generator")

    for scenario_name, (base_seed, _) in SCENARIOS.items():
        mnar_strength, mar_strength = get_scenario_params(scenario_name)
        print(f"\n{'='*60}")
        print(f"Scenario : {scenario_name}")
        print(f"MNAR     : {mnar_strength}  MAR: {mar_strength}")

        for cohort_seed in COHORT_SEEDS:
            gen_seed = base_seed * 1000 + cohort_seed
            print(f"\n  Cohort seed {cohort_seed} (gen_seed={gen_seed})...")

            df = simulate_dataset(
                seed=gen_seed,
                mnar_strength=mnar_strength,
                mar_strength=mar_strength,
            )

            print(f"  Raw: {df['RANDID'].nunique()} patients, {len(df)} rows")

            # Quick sanity check
            last_time = df.groupby("RANDID")["TIME"].max()
            timedth   = df.groupby("RANDID")["TIMEDTH"].first()
            gap_bins  = (
                np.ceil(timedth / 10) - np.floor(last_time / 10)
            ).astype(int)
            print(f"  Gap bins — min: {gap_bins.min()}  "
                  f"median: {gap_bins.median():.0f}  "
                  f"mean: {gap_bins.mean():.1f}")
            bad_gap = (gap_bins <= MIN_GAP_BINS).sum()
            if bad_gap > 0:
                print(f"  ⚠️  {bad_gap} patients with gap_bins <= {MIN_GAP_BINS}")

            out_dir = os.path.join(
                OUT_ROOT, scenario_name, f"cohortseed_{cohort_seed:03d}"
            )
            split_and_save(df, out_dir, cohort_seed, scenario_name,
                          mnar_strength, mar_strength, gen_seed)

    print(f"\n{'='*60}")
    print("Done. Data saved to:", OUT_ROOT)
    print("="*60)
