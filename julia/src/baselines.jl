"""Baseline change detection: log-ratio + K&I, CFAR, dan signifikansi."""

using Distributions
using Statistics

"""Total power rata-rata dari diagonal kovarians."""
function total_power(C::Array{ComplexF64,4})
    P = (real.(C[:, :, 1, 1]) .+ real.(C[:, :, 2, 2]) .+ real.(C[:, :, 3, 3])) ./ 3.0
    return P
end

"""DI log-ratio |log10(P1/P2)| pada total power, dinormalisasi min-max."""
function logratio_di(td)
    P1 = total_power(td.C1)
    P2 = total_power(td.C2)
    lr = abs.(log10.(max.(P1, 1e-30) ./ max.(P2, 1e-30)))
    vals = lr[td.mask]
    lo, hi = extrema(vals)
    norm = Array{Float32}(undef, size(lr))
    if hi > lo
        @. norm = clamp((lr - lo) / (hi - lo), 0.0, 1.0)
    else
        fill!(norm, 0.0f0)
    end
    norm[.!td.mask] .= 0.0f0
    return (di=lr, di_norm=norm, lo=lo, hi=hi)
end

"""CFAR: fit Gamma pada clutter (kuantil bawah), threshold pada 1-pfa."""
function cfar_map(di_norm, mask::BitMatrix; pfa::Float64=1e-2, fit_quantile::Float64=0.99)
    vals = Float64.(di_norm[mask])
    cutoff = quantile(vals, fit_quantile)
    clutter = vals[(vals .> 0) .& (vals .<= cutoff)]
    thr = NaN
    if length(clutter) > 100
        try
            g = fit(Gamma, clutter)
            thr = quantile(g, 1 - pfa)
        catch
            thr = quantile(vals, 1 - pfa)
        end
    else
        thr = quantile(vals, 1 - pfa)
    end
    thr = clamp(thr, 0.0, 1.0)
    change = (di_norm .> thr) .& mask
    return (threshold=thr, change=change, model="gamma-clutter", pfa=pfa,
            fit_quantile=fit_quantile)
end

"""Threshold signifikansi alpha via kuantil empiris DI."""
function significance_map(di_norm, mask::BitMatrix; alpha::Float64=0.05)
    vals = Float64.(di_norm[mask])
    thr = quantile(vals, 1 - alpha)
    change = (di_norm .> thr) .& mask
    return (threshold=thr, change=change, model="empirical-quantile", alpha=alpha)
end
