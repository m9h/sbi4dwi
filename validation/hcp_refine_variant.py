#!/usr/bin/env python3
"""Refit the plus-x refinement on both HCP visits with a model variant, then f_i / fixel scan–rescan metrics against the default.
    uv run python validation/hcp_refine_variant.py --subject 105923 --variant noiso
variants: noiso (CSF/GM off, B3's dense-WM model), csfonly (GM ball off), k2 (two fibres)."""
import argparse, json, sys, time
from dataclasses import replace
import numpy as np, jax
sys.path.insert(0, "validation"); import validate_hcp_retest as V
from dmipy_jax.validation import dipy_refine as dr, prism_jax as pj
from dipy.data import default_sphere

ap = argparse.ArgumentParser(); ap.add_argument("--subject", default="105923"); ap.add_argument("--variant", default="noiso"); ap.add_argument("--n-iter", type=int, default=300)
a = ap.parse_args(); V.SUBJ = a.subject; V.OUT = V.ROOT / "b1" / a.subject
K = 2 if a.variant == "k2" else 3
for session in ["test", "retest"]:
    out = V.OUT / session / f"refine_{a.variant}.npz"
    if out.exists(): continue
    data, affine, gtab, mask, aparc = V.load_session(session); z = np.load(V.OUT / session / "msmt.npz")
    cfg = dr.default_config(K, n_iter=a.n_iter, wm_only=(a.variant == "noiso"))
    if a.variant == "csfonly": cfg = replace(cfg, use_gm=False) if hasattr(cfg, "use_gm") else cfg
    pam = V.build_pam(z["dirs"], z["vals"], mask, affine, default_sphere)
    t0 = time.time(); _, _, fit = dr.refine_peaks(data, gtab, mask, pam, n_fibres=K, cfg=cfg, affine=affine, laplace=False); jax.block_until_ready(fit.fintra)
    np.savez_compressed(out, dirs=fit.dirs.astype(np.float32), wm_fracs=fit.wm_fracs.astype(np.float32), fracs=fit.fracs.astype(np.float32), fintra=fit.fintra.astype(np.float32), d_par=fit.d_par)
    V.log(f"{session} {a.variant}: {time.time()-t0:.0f}s D∥={fit.d_par*1e9:.2f} f_i(WM)={fit.fintra[np.isin(aparc, V.WM_ONLY)[mask]].mean():.3f}")
    del data
# metrics
T = np.load(V.OUT / "test" / "grid.npz"); R = np.load(V.OUT / "retest" / "grid.npz")
reg = np.array(json.load(open(V.OUT / "compare_results.json"))["registration"]["affine"]); lin = V.resample_to_test(reg, T, R); ok = lin >= 0
def pull(x): y = np.zeros((len(lin),) + x.shape[1:], x.dtype); y[ok] = x[lin[ok]]; return y
sel = ok & T["wm"][T["mask"]] & pull(R["wm"][R["mask"]]); Rrot = reg[:3, :3] / np.cbrt(np.linalg.det(reg[:3, :3]))
res = {}
for name in ["refine", f"refine_{a.variant}"]:
    t, r = np.load(V.OUT / "test" / f"{name}.npz"), np.load(V.OUT / "retest" / f"{name}.npz")
    x, y = t["fintra"][sel].astype(float), pull(r["fintra"])[sel].astype(float); d = x - y
    kt, kr = t["wm_fracs"] >= 0.1, pull(r["wm_fracs"] >= 0.1); both = sel & (kt.sum(1) > 0) & (kr.sum(1) > 0)
    ang = V.angle_deg(t["dirs"][both, 0], (pull(r["dirs"]) @ Rrot)[both, 0])
    fiso = t["fracs"][:, 0] + t["fracs"][:, 1]
    res[name] = {"ccc": V.ccc(x, y), "mean_abs_diff": float(np.abs(d).mean()), "wscv": float(np.sqrt(np.mean(d ** 2) / 2) / np.mean((x + y) / 2)), "mean": float(x.mean()), "f_iso_wm_mean": float(fiso[sel].mean()),
                 "main_median_deg": float(np.median(ang)), "main_within10": float((ang < 10).mean()), "count_agree": float((kt.sum(1)[sel] == kr.sum(1)[sel]).mean()), "mean_count": float(kt.sum(1)[sel].mean())}
    o = res[name]; print(f"{name:14s} f_i CCC={o['ccc']:.3f} |Δ|={o['mean_abs_diff']:.3f} wsCV={100*o['wscv']:.1f}% mean {o['mean']:.3f} (f_iso WM {o['f_iso_wm_mean']:.3f}) | main fixel Δθ median {o['main_median_deg']:.2f}° <10° {100*o['main_within10']:.1f}% | count agree {100*o['count_agree']:.1f}% (mean {o['mean_count']:.2f})")
json.dump(res, open(V.OUT / f"variant_{a.variant}.json", "w"), indent=1)
