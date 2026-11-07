#!/usr/bin/env python3
"""Kittler-Illingworth style threshold search with a posterior-based cost.

J(T)   = sum_d h(d) c(d,T)
c(d,T) = -2 ln P(w_u|d)  if d <= T   (unchanged)
         -2 ln P(w_c|d)  if d >  T   (changed)
P(w_k|d) = pi_k f_k(d) / (pi_u f_u(d) + pi_c f_c(d))

f_k is Gaussian / Weibull / Gamma, fitted by (histogram-weighted) MLE.

Usage:
    python ki_threshold.py --input DI.npy --pdf weibull [--L 256] [--save-map map.npy]
"""
import argparse
import numpy as np
from scipy import stats, optimize, special

EPS = 1e-6      # replaces level 0 for weibull / gamma (undefined at 0)
TINY = 1e-300


# ----------------------------------------------------------------- loading
def load_levels(path, L):
    """Load normalized DI (values ~ dl/L) -> integer levels 0..L-1."""
    x = np.load(path, allow_pickle=False).astype(np.float64)
    lv = np.clip(np.rint(x * L), 0, L - 1).astype(np.int64)
    return x, lv


# ----------------------------------------------------------------- fitting
def _support(levels, pdf):
    """Level values used by the PDF (level 0 -> EPS for positive-support PDFs)."""
    d = levels.astype(np.float64)
    if pdf in ("weibull", "gamma"):
        d = np.where(d <= 0, EPS, d)
    return d


def wmean_var(d, w):
    W = w.sum()
    m = (w * d).sum() / W
    v = (w * (d - m) ** 2).sum() / W
    return m, v


def fit_gaussian(d, w):
    m, v = wmean_var(d, w)
    return {"mu": m, "sigma": max(np.sqrt(v), 1e-6)}


def fit_gamma(d, w):
    """Weighted MLE, loc=0. Solve ln k - psi(k) = ln(mean) - mean(ln d)."""
    W = w.sum()
    m = (w * d).sum() / W
    s = np.log(m) - (w * np.log(d)).sum() / W
    s = max(s, 1e-12)
    # Minka's initial guess, then Newton-ish root find
    k0 = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    f = lambda k: np.log(k) - special.digamma(k) - s
    try:
        k = optimize.brentq(f, k0 / 10, k0 * 10 + 1, xtol=1e-12)
    except ValueError:
        k = k0
    scale = m / k
    return {"shape": k, "scale": scale, "rate": 1.0 / scale}


def fit_weibull(d, w):
    """Weighted MLE, loc=0. Solve for shape k, then scale."""
    W = w.sum()
    dmax = d.max()
    d0, d = d, d / dmax              # rescale to avoid overflow in d**k
    ln = np.log(d)
    mean_ln = (w * ln).sum() / W

    def g(k):
        dk = d ** k
        return (w * dk * ln).sum() / (w * dk).sum() - 1.0 / k - mean_ln

    lo, hi = 1e-2, 1.0
    while g(hi) < 0 and hi < 1e3:
        hi *= 2
    try:
        k = optimize.brentq(g, lo, hi, xtol=1e-12)
    except ValueError:
        k = 1.0
    lam = ((w * d ** k).sum() / W) ** (1.0 / k) * dmax
    return {"shape": k, "scale": lam}


FIT = {"gaussian": fit_gaussian, "weibull": fit_weibull, "gamma": fit_gamma}


def logpdf(pdf, params, d):
    if pdf == "gaussian":
        return stats.norm.logpdf(d, params["mu"], params["sigma"])
    if pdf == "weibull":
        return stats.weibull_min.logpdf(d, params["shape"], 0, params["scale"])
    return stats.gamma.logpdf(d, params["shape"], 0, params["scale"])


# -------------------------------------------------------------------- core
def evaluate_T(T, h, pdf, L, detail=False, cost_mode="posterior"):
    """Return J(T) (inf if infeasible). h = normalized histogram (len L)."""
    levels = np.arange(L)
    hu, hc = h[: T + 1], h[T + 1:]
    pi_u, pi_c = hu.sum(), hc.sum()
    # need two classes with >=2 occupied bins each to fit parameters
    if (hu > 0).sum() < 2 or (hc > 0).sum() < 2:
        return np.inf, None

    du = _support(levels[: T + 1], pdf)
    dc = _support(levels[T + 1:], pdf)
    try:
        pu = FIT[pdf](du[hu > 0], hu[hu > 0])
        pc = FIT[pdf](dc[hc > 0], hc[hc > 0])
    except Exception:
        return np.inf, None

    d_all = _support(levels, pdf)
    lu = np.log(max(pi_u, TINY)) + logpdf(pdf, pu, d_all)
    lc = np.log(max(pi_c, TINY)) + logpdf(pdf, pc, d_all)
    denom = np.logaddexp(lu, lc)
    log_post_u, log_post_c = lu - denom, lc - denom

    if cost_mode == "posterior":      # as specified: -2 ln P(w|d)
        cost = np.where(levels <= T, -2 * log_post_u, -2 * log_post_c)
    else:                              # classic KI: -2 ln [pi_k f_k(d)]
        cost = np.where(levels <= T, -2 * lu, -2 * lc)
    J = float((h * cost).sum())
    if not np.isfinite(J):
        return np.inf, None
    if detail:
        return J, dict(pi_u=pi_u, pi_c=pi_c, pu=pu, pc=pc, du=du, dc=dc,
                       hu=hu, hc=hc, lu=lu, lc=lc, log_post_u=log_post_u,
                       log_post_c=log_post_c, cost=cost, denom=denom)
    return J, None


# ------------------------------------------------------------------ output
def fmt_params(pdf, p):
    if pdf == "gaussian":
        return f"mu = {p['mu']:.6f}, sigma = {p['sigma']:.6f}"
    if pdf == "weibull":
        return f"shape (k) = {p['shape']:.6f}, scale (lambda) = {p['scale']:.6f}"
    return (f"shape (k) = {p['shape']:.6f}, scale (theta) = {p['scale']:.6f}, "
            f"rate (1/theta) = {p['rate']:.6f}")


def verbose_report(T, h, pdf, L, x, lv, cost_mode="posterior"):
    J, r = evaluate_T(T, h, pdf, L, detail=True, cost_mode=cost_mode)
    levels = np.arange(L)
    sep = "=" * 72
    print(f"\n{sep}\nSTEP-BY-STEP RECALCULATION AT OPTIMAL T = {T}  "
          f"(original scale: {T / L:.6f}) | pdf = {pdf}\n{sep}")

    print(f"\n[1] Priors   pi_u = P(dl <= T) = {r['pi_u']:.6f}   "
          f"pi_c = P(dl > T) = {r['pi_c']:.6f}")

    def cls_stats(lvls, hh):
        m, v = wmean_var(lvls.astype(float), hh)
        return m, v, np.sqrt(v)

    mu_u, var_u, sd_u = cls_stats(levels[: T + 1], r["hu"])
    mu_c, var_c, sd_c = cls_stats(levels[T + 1:], r["hc"])

    print("\n[2] Class statistics (level domain 0..L-1, original = level / L)")
    print(f"    unchanged (dl <= T): mean = {mu_u:.6f} (orig {mu_u / L:.6f}) | "
          f"std = {sd_u:.6f} (orig {sd_u / L:.6f}) | var = {var_u:.6f} (orig {var_u / L**2:.8f})")
    print(f"    changed   (dl >  T): mean = {mu_c:.6f} (orig {mu_c / L:.6f}) | "
          f"std = {sd_c:.6f} (orig {sd_c / L:.6f}) | var = {var_c:.6f} (orig {var_c / L**2:.8f})")

    print(f"\n[3] MLE parameters of {pdf} (level domain)")
    print(f"    unchanged: {fmt_params(pdf, r['pu'])}")
    print(f"    changed  : {fmt_params(pdf, r['pc'])}")

    # evaluation points: the class means (nearest level for the posterior terms)
    print("\n[4] PDF values evaluated at the class means")
    pts = [("mean of unchanged", mu_u), ("mean of changed", mu_c)]
    for name, m in pts:
        dm = m if pdf == "gaussian" else max(m, EPS)
        fu = float(np.exp(logpdf(pdf, r["pu"], dm)))
        fc = float(np.exp(logpdf(pdf, r["pc"], dm)))
        print(f"    d = {m:.4f} ({name}, orig {m / L:.6f}):  "
              f"f_u(d) = {fu:.6e}   f_c(d) = {fc:.6e}")

    print("\n[5] Cost function")
    for name, m in pts:
        i = int(np.clip(round(m), 0, L - 1))
        pu_ = float(np.exp(r["log_post_u"][i]))
        pc_ = float(np.exp(r["log_post_c"][i]))
        side = "u" if i <= T else "c"
        print(f"    level {i:3d} ({name}): P(w_u|d) = {pu_:.6f}  P(w_c|d) = {pc_:.6f}  "
              f"-> c(d,T) = -2 ln P(w_{side}|d) = {r['cost'][i]:.6f}  h(d) = {h[i]:.6e}")
    Ju = float((h[: T + 1] * r["cost"][: T + 1]).sum())
    Jc = float((h[T + 1:] * r["cost"][T + 1:]).sum())
    print(f"    sum over d <= T : {Ju:.6f}")
    print(f"    sum over d >  T : {Jc:.6f}")
    print(f"    J(T) = {J:.6f}")

    n = lv.size
    n_c = int((lv > T).sum())
    print("\n[6] Threshold result")
    print(f"    optimal T = {T} (level)  = {T / L:.6f} (original scale)")
    print(f"    rule: dl <= {T} -> unchanged, dl > {T} -> changed")
    print(f"    unchanged pixels: {n - n_c} ({100 * (n - n_c) / n:.2f}%) | "
          f"changed pixels: {n_c} ({100 * n_c / n:.2f}%)")
    return J


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".npy normalized difference image")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--pdf", choices=list(FIT), required=True)
    ap.add_argument("--cost", choices=["posterior", "likelihood"], default="posterior",
                    help="posterior = your formulas (default); likelihood = classic KI -2 ln[pi f]")
    ap.add_argument("--save-map", help="optional .npy path for the binary change map")
    a = ap.parse_args()

    x, lv = load_levels(a.input, a.L)
    print(f"Image {x.shape}, original range [{x.min():.6f}, {x.max():.6f}] "
          f"-> levels [{lv.min()}, {lv.max()}], L = {a.L}, pdf = {a.pdf}")

    h = np.bincount(lv.ravel(), minlength=a.L).astype(np.float64)
    h /= h.sum()

    # pass 1: brute force
    Js = np.full(a.L, np.inf)
    for T in range(a.L):
        Js[T], _ = evaluate_T(T, h, a.pdf, a.L, cost_mode=a.cost)
    if not np.isfinite(Js).any():
        raise SystemExit("No feasible threshold found.")
    T_opt = int(np.argmin(Js))
    print(f"\nPass 1 (brute force over T = 0..{a.L - 1}): "
          f"T_opt = {T_opt}, J_min = {Js[T_opt]:.6f}")

    # pass 2: verbose recalculation
    J2 = verbose_report(T_opt, h, a.pdf, a.L, x, lv, a.cost)
    print(f"\nCheck: pass-1 J = {Js[T_opt]:.10f} | pass-2 J = {J2:.10f} | "
          f"diff = {abs(Js[T_opt] - J2):.2e}")

    if a.save_map:
        np.save(a.save_map, (lv > T_opt).astype(np.uint8))
        print(f"Change map saved to {a.save_map}")


if __name__ == "__main__":
    main()
