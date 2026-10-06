#!/usr/bin/env python3
"""Does connectome edge uncertainty help group statistics? Between-subject ICC(3,1) of test vs retest edge weights across the clean cohort,
for count connectomes (MSMT, refine), the posterior mean, CV-weighted and CV-thresholded variants, and edges stratified by posterior CV.
    uv run python validation/uncertainty_group_reliability.py"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/data/datasets/hcp/b1"); subs = json.load(open("validation/hcp_retest/cohort_groups.json"))["clean"]
iu = np.triu_indices(68, 1)

def load(s, v):
    z = np.load(ROOT / s / v / "connectomes.npz"); P = z["posterior"][:, iu[0], iu[1]].astype(float)
    return {"msmt": z["msmt"][iu].astype(float), "refine": z["refine"][iu].astype(float), "post": P}

D = {s: {v: load(s, v) for v in ["test", "retest"]} for s in subs}

def feats(c):
    m = c["post"].mean(0); cv = c["post"].std(0) / np.maximum(m, 1e-9)
    return {"msmt": np.log1p(c["msmt"]), "refine": np.log1p(c["refine"]), "post_mean": np.log1p(m),
            "post_cv_weighted": np.log1p(m) * (1 - np.minimum(cv, 1)), "post_cv<0.5": np.where(cv < 0.5, np.log1p(m), 0.0)}, cv

F = {s: {v: feats(D[s][v]) for v in ["test", "retest"]} for s in subs}

def icc31(x, y):  # x,y (n_subj,)
    d = np.stack([x, y], 1); k = 2; n = len(x)
    gm = d.mean(); ms_r = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_e = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)

names = list(F[subs[0]]["test"][0].keys())
X = {n: np.stack([F[s]["test"][0][n] for s in subs]) for n in names}; Y = {n: np.stack([F[s]["retest"][0][n] for s in subs]) for n in names}
present = (np.stack([np.log1p(D[s]["test"]["post"].mean(0)) for s in subs]) > 0).mean(0) >= 0.5   # edges present in >=50% of visits
cv_edge = np.nanmean(np.stack([F[s]["test"][1] for s in subs]), 0)
res = {"n_subjects": len(subs), "n_edges_tested": int(present.sum())}
print(f"{len(subs)} subjects, {present.sum()} edges present in ≥50% of test visits")
for n in names:
    icc = np.array([icc31(X[n][:, e], Y[n][:, e]) if X[n][:, e].std() > 0 and Y[n][:, e].std() > 0 else np.nan for e in np.where(present)[0]])
    res[n] = {"median_icc": float(np.nanmedian(icc)), "mean_icc": float(np.nanmean(icc)), "frac_icc>0.6": float(np.nanmean(icc > 0.6))}
    print(f"{n:18s} median ICC {np.nanmedian(icc):.3f}  mean {np.nanmean(icc):.3f}  frac>0.6 {np.nanmean(icc > 0.6):.2f}")
# stratify by mean posterior CV (edge-wise, averaged over subjects)
idx = np.where(present)[0]; icc = np.array([icc31(X["post_mean"][:, e], Y["post_mean"][:, e]) for e in idx]); cvs = cv_edge[idx]
q = np.nanquantile(cvs, [1 / 3, 2 / 3]); res["post_mean_icc_by_cv_tertile"] = {}
for lab, sel in [("low CV", cvs <= q[0]), ("mid CV", (cvs > q[0]) & (cvs <= q[1])), ("high CV", cvs > q[1])]:
    res["post_mean_icc_by_cv_tertile"][lab] = {"median_icc": float(np.nanmedian(icc[sel])), "mean_cv": float(np.nanmean(cvs[sel])), "n": int(sel.sum())}
    print(f"posterior-mean ICC, {lab:8s} (CV {np.nanmean(cvs[sel]):.2f}, n={sel.sum()}): median {np.nanmedian(icc[sel]):.3f}")
from scipy.stats import spearmanr
rho = spearmanr(cvs, icc, nan_policy="omit"); res["spearman_edgeCV_vs_ICC"] = [float(rho.statistic), float(rho.pvalue)]; print(f"Spearman(edge CV, ICC) = {rho.statistic:.3f} p={rho.pvalue:.2g}")
json.dump(res, open("validation/hcp_retest/uncertainty_group_reliability.json", "w"), indent=1)
