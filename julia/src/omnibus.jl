"""DI omnibus likelihood-ratio test statistic (Eq. 10) dan normalisasi."""

using LinearAlgebra

function det_positive(M)
    d = real(det(M))
    return max(d, 1e-30)
end

"""
    omnibus_raw(C1, C2; p, m, n)

Nilai Eq. 10 sebelum orientasi/normalisasi. Untuk n=2 berlaku:
    d = (2p^2 - 4pm - 1)/(2p) * (p n ln n + ln|C1| + ln|C2| - n ln|C1+C2|)
"""
function omnibus_raw(C1::Array{ComplexF64,4}, C2::Array{ComplexF64,4};
                     p::Int=3, m::Int=189, n::Int=2)
    coeff = (2.0 * p^2 - 4.0 * p * m - 1.0) / (2.0 * p)
    h, w = size(C1, 1), size(C1, 2)
    out = Array{Float64}(undef, h, w)
    const_term = p * n * log(n)
    for j in 1:w, i in 1:h
        M1 = Matrix(@view C1[i, j, :, :])
        M2 = Matrix(@view C2[i, j, :, :])
        d1 = det_positive(M1)
        d2 = det_positive(M2)
        ds = det_positive(M1 + M2)
        bracket = const_term + log(d1) + log(d2) - n * log(ds)
        out[i, j] = coeff * bracket
    end
    return out
end

"""Min-max normalisasi pada piksel valid; mengembalikan (di, di_norm, lo, hi)."""
function normalize_di(raw, mask; orientation::AbstractString="pos")
    di = orientation == "neg" ? -raw : raw
    vals = di[mask]
    isempty(vals) && error("Tidak ada piksel valid")
    lo, hi = minimum(vals), maximum(vals)
    norm = Array{Float32}(undef, size(di))
    if hi > lo
        @. norm = clamp((di - lo) / (hi - lo), 0.0, 1.0)
    else
        fill!(norm, 0.0f0)
    end
    norm[.!mask] .= 0.0f0
    return di, norm, lo, hi
end

function compute_di(td; p::Int=3, m::Int=189, n::Int=2, orientation::AbstractString="pos")
    raw = omnibus_raw(td.C1, td.C2; p=p, m=m, n=n)
    di, di_norm, lo, hi = normalize_di(raw, td.mask; orientation=orientation)
    return (raw=raw, di=di, di_norm=di_norm, lo=lo, hi=hi, mask=td.mask)
end
