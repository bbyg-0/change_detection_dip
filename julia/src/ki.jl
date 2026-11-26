"""Kittler–Illingworth thresholding untuk DI (Gaussian/Weibull/gamma).

Port dari src/ki_threshold.py (mode likelihood/posterior) dengan pengaman
min_bins agar MLE tidak degenerasi pada kelas dengan sedikit bin.
"""

using Distributions
using SpecialFunctions
using Roots

const EPS_KI = 1e-6
const TINY_KI = 1e-300
const MIN_BINS_KI = 5

function logaddexp(a::Float64, b::Float64)
    m = max(a, b)
    return m + log1p(exp(-abs(a - b)))
end

function support(levels::AbstractVector, pdf::AbstractString)
    if pdf == "gaussian"
        return Float64.(levels)
    end
    return [x <= 0 ? EPS_KI : Float64(x) for x in levels]
end

function wmean_var(d, w)
    W = sum(w)
    m = sum(w .* d) / W
    v = sum(w .* (d .- m) .^ 2) / W
    return m, v
end

fit_gaussian(d, w) = begin
    m, v = wmean_var(d, w)
    (mu=m, sigma=max(sqrt(v), 1e-6))
end

function fit_gamma(d, w)
    W = sum(w)
    m = sum(w .* d) / W
    s = log(m) - sum(w .* log.(d)) / W
    s = max(s, 1e-12)
    k0 = (3 - s + sqrt((s - 3)^2 + 24s)) / (12s)
    f(k) = log(k) - digamma(k) - s
    k = try
        find_zero(f, (k0 / 10, k0 * 10 + 1), Roots.Brent())
    catch
        k0
    end
    scale = m / k
    return (shape=k, scale=scale, rate=1.0 / scale)
end

function fit_weibull(d, w)
    W = sum(w)
    dmax = maximum(d)
    dn = d ./ dmax
    ln = log.(dn)
    mean_ln = sum(w .* ln) / W
    g(k) = sum(w .* (dn .^ k) .* ln) / sum(w .* (dn .^ k)) - 1.0 / k - mean_ln
    lo, hi = 1e-2, 1.0
    while g(hi) < 0 && hi < 1e3
        hi *= 2
    end
    k = try
        find_zero(g, (lo, hi), Roots.Brent())
    catch
        1.0
    end
    lam = (sum(w .* (dn .^ k)) / W)^(1.0 / k) * dmax
    return (shape=k, scale=lam)
end

const FIT = Dict("gaussian" => fit_gaussian, "weibull" => fit_weibull, "gamma" => fit_gamma)

function logpdf_pdf(pdf::AbstractString, params, d)
    if pdf == "gaussian"
        return logpdf.(Normal(params.mu, params.sigma), d)
    elseif pdf == "weibull"
        return logpdf.(Weibull(params.shape, params.scale), d)
    else
        return logpdf.(Gamma(params.shape, params.scale), d)
    end
end

function evaluate_T(T::Int, lobes, hobs, dsup, pdf::AbstractString;
                    cost_mode::AbstractString="likelihood", min_bins::Int=MIN_BINS_KI)
    sel_u = lobes .<= T
    nu = count(sel_u)
    (nu < min_bins || (length(lobes) - nu) < min_bins) && return Inf, nothing

    du, wu = dsup[sel_u], hobs[sel_u]
    dc, wc = dsup[.!sel_u], hobs[.!sel_u]
    try
        pu = FIT[pdf](du, wu)
        pc = FIT[pdf](dc, wc)
        pi_u = sum(wu)
        pi_c = sum(wc)
        lu = log(max(pi_u, TINY_KI)) .+ logpdf_pdf(pdf, pu, dsup)
        lc = log(max(pi_c, TINY_KI)) .+ logpdf_pdf(pdf, pc, dsup)
        if cost_mode == "posterior"
            denom = logaddexp.(lu, lc)
            cost = ifelse.(sel_u, -2 .* (lu .- denom), -2 .* (lc .- denom))
        else
            cost = ifelse.(sel_u, -2 .* lu, -2 .* lc)
        end
        J = sum(hobs .* cost)
        if !isfinite(J)
            return Inf, nothing
        end
        return Float64(J), (pu=pu, pc=pc, pi_u=pi_u, pi_c=pi_c)
    catch
        return Inf, nothing
    end
end

function ki_threshold(di_norm, mask::BitMatrix; pdf::AbstractString="gamma", L::Int=2048,
                      cost_mode::AbstractString="likelihood", min_bins::Int=MIN_BINS_KI)
    x = Float64.(di_norm)
    lv = clamp.(round.(Int, x .* L), 0, L - 1)
    h = zeros(Float64, L)
    for v in lv[mask]
        h[v + 1] += 1.0
    end
    h ./= sum(h)

    occ = findall(>(0.0), h)
    lobes = Float64.(occ .- 1)
    hobs = h[occ]
    dsup = support(lobes, pdf)

    Js = fill(Inf, L)
    bestJ = Inf
    bestT = -1
    bestPar = nothing
    for T in 0:(L - 1)
        J, par = evaluate_T(T, lobes, hobs, dsup, pdf;
                            cost_mode=cost_mode, min_bins=min_bins)
        Js[T + 1] = J
        if J < bestJ
            bestJ = J
            bestT = T
            bestPar = par
        end
    end
    bestT < 0 && error("Tidak ada threshold yang feasible")

    change = (lv .> bestT) .& mask
    return (T=bestT, J=bestJ, Js=Js, change=change, levels=lv, hist=h,
            params=bestPar, pdf=pdf, L=L, cost_mode=cost_mode, min_bins=min_bins)
end
