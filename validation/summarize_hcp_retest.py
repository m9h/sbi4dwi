#!/usr/bin/env python3
"""Aggregate validate_hcp_retest.py compare results over subjects → mean ± sd table + JSON.

    uv run python validation/summarize_hcp_retest.py [--root /data/datasets/hcp/b1]
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/data/datasets/hcp/b1"); ap.add_argument("--out", default="validation/hcp_retest/summary.json")
    a = ap.parse_args()
    subs = sorted(p.parent.name for p in Path(a.root).glob("*/compare_results.json"))
    R = {s: json.load(open(Path(a.root) / s / "compare_results.json")) for s in subs}
    print(f"{len(subs)} subjects: {' '.join(subs)}\n")
    rows = []
    def add(name, getter):
        vals = []
        for s in subs:
            try: vals.append(float(getter(R[s])))
            except (KeyError, TypeError): pass
        if vals: rows.append((name, np.mean(vals), np.std(vals), len(vals)))
    for k, lab in [("refine_fi", "f_i refine"), ("force_nd", "FORCE ND"), ("force_nd_wm", "FORCE ND·f_wm"), ("noddi_ndi", "NODDI NDI"), ("noddi_ndi_tissue", "NODDI NDI·(1−FWF)"), ("noddi_odi", "NODDI ODI"), ("fa", "DTI FA")]:
        add(f"{lab}: CCC", lambda r, k=k: r["scalars"][k]["ccc"]); add(f"{lab}: |Δ|", lambda r, k=k: r["scalars"][k]["mean_abs_diff"]); add(f"{lab}: wsCV %", lambda r, k=k: 100 * r["scalars"][k]["wscv"])
    for k in ["msmt", "refine", "force"]:
        add(f"{k}: main fixel Δθ median °", lambda r, k=k: r["fixels"][k]["main_median_deg"]); add(f"{k}: main fixel <10° %", lambda r, k=k: 100 * r["fixels"][k]["main_within10"])
        add(f"{k}: fixel count agree %", lambda r, k=k: 100 * r["fixels"][k]["count_agree"])
        add(f"{k}: connectome r(log)", lambda r, k=k: r["connectomes"][k]["r_log"]); add(f"{k}: connectome Dice", lambda r, k=k: r["connectomes"][k]["dice"])
    add("posterior mean: connectome r(log)", lambda r: r["connectomes"]["posterior_mean"]["r_log"]); add("posterior mean: connectome Dice", lambda r: r["connectomes"]["posterior_mean"]["dice"])
    for cv in ["0.5", "0.3"]:
        add(f"posterior CV<{cv}: r(log)", lambda r, cv=cv: r["connectomes"][f"posterior_cv{cv}"]["r_log"])
    add("edge CV vs disagreement: Spearman", lambda r: r["connectomes"]["edge_cv_vs_disagreement_spearman"])
    add("calibration: Spearman(σ, Δθ)", lambda r: r["calibration"]["spearman_sigma_vs_dtheta"]); add("calibration: obs/pred median", lambda r: r["calibration"]["obs_over_pred_median"])
    add("calibration: within predicted 90% %", lambda r: 100 * r["calibration"]["frac_within_pred_p90"])
    Vn = {s: json.load(open(Path(a.root) / s / "variant_noiso.json")) for s in subs if (Path(a.root) / s / "variant_noiso.json").exists()}
    if Vn:
        for k, lab in [("refine_noiso", "f_i refine no-iso"), ("refine", "f_i refine (variant run)")]:
            for m, n in [("ccc", "CCC"), ("mean_abs_diff", "|Δ|"), ("wscv", "wsCV %"), ("main_median_deg", "main fixel Δθ median °")]:
                vals = [Vn[s][k][m] * (100 if m == "wscv" else 1) for s in Vn if k in Vn[s]]
                if vals: rows.append((f"{lab}: {n}", np.mean(vals), np.std(vals), len(vals)))
    add("registration rotation °", lambda r: r["registration"]["rotation_deg"]); add("WM Dice", lambda r: r["wm_dice"])
    w = max(len(n) for n, *_ in rows)
    for n, m, sd, k in rows:
        print(f"{n:{w}s}  {m:7.3f} ± {sd:.3f}  (n={k})")
    json.dump({"subjects": subs, "rows": [{"name": n, "mean": m, "sd": sd, "n": k} for n, m, sd, k in rows], "per_subject": R}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
