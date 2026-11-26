"""Membaca prep.npz (hasil Python) dan menyusun matriks kovarians PolSAR."""

using NPZ
using JSON3
using LinearAlgebra

struct TileData
    C1::Array{ComplexF64,4}
    C2::Array{ComplexF64,4}
    mask::BitMatrix
    meta::Any
end

function build_covariance(z, ds::AbstractString)
    hh = Float64.(z["$(ds)_HHHH"])
    hv = Float64.(z["$(ds)_HVHV"])
    vv = Float64.(z["$(ds)_VVVV"])
    c12 = sqrt(2.0) .* ComplexF64.(z["$(ds)_HHHV"])
    c13 = ComplexF64.(z["$(ds)_HHVV"])
    c23 = sqrt(2.0) .* ComplexF64.(z["$(ds)_HVVV"])

    s = size(hh)
    C = Array{ComplexF64}(undef, s..., 3, 3)
    @views C[:, :, 1, 1] .= hh
    @views C[:, :, 2, 2] .= 2.0 .* hv
    @views C[:, :, 3, 3] .= vv
    @views C[:, :, 1, 2] .= c12
    @views C[:, :, 2, 1] .= conj.(c12)
    @views C[:, :, 1, 3] .= c13
    @views C[:, :, 3, 1] .= conj.(c13)
    @views C[:, :, 2, 3] .= c23
    @views C[:, :, 3, 2] .= conj.(c23)
    return C
end

function diagonal_loading!(C::Array{ComplexF64,4}, eps::Float64)
    eps <= 0 && return C
    p = size(C, 3)
    tr = zeros(Float64, size(C, 1), size(C, 2))
    for i in 1:p
        @views tr .+= real.(C[:, :, i, i])
    end
    tr ./= p
    for i in 1:p
        @views C[:, :, i, i] .+= eps .* tr
    end
    return C
end

function is_psd(M; tol=1e-8)
    A = Matrix(M)  # view 3x3 tidak kontigu; salin dulu
    e = eigvals(Hermitian(A))
    s = max(abs(real(tr(A))) / 3, 1e-30)
    return e[1] >= -tol * s
end

function psd_mask!(C1, C2, mask::BitMatrix)
    out = copy(mask)
    for idx in findall(mask)
        i, j = idx[1], idx[2]
        if !(is_psd(@view C1[i, j, :, :]) && is_psd(@view C2[i, j, :, :]))
            out[i, j] = false
        end
    end
    return out
end

function load_tile(prep_npz::AbstractString, meta_json::AbstractString; eps::Float64=1e-6)
    z = npzread(prep_npz)
    meta = JSON3.read(read(meta_json, String))
    C1 = build_covariance(z, "T1")
    C2 = build_covariance(z, "T5")
    diagonal_loading!(C1, eps)
    diagonal_loading!(C2, eps)
    mask = BitMatrix(z["mask"] .!= 0)
    mask = psd_mask!(C1, C2, mask)
    return TileData(C1, C2, mask, meta)
end
