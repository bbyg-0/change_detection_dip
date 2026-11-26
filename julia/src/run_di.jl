"""Hitung DI untuk satu tile (prep.npz) dan simpan hasilnya."""

include("UAVSARChangeDetection.jl")
using .UAVSARChangeDetection
using NPZ, JSON3, Printf

function main(args)
    isempty(args) && error("Usage: julia run_di.jl <prep.npz> <meta.json> <out_prefix> [orientation]")
    prep_npz = args[1]
    meta_json = args[2]
    out_prefix = args[3]
    orientation = length(args) >= 4 ? args[4] : "pos"

    t0 = time()
    td = load_tile(prep_npz, meta_json)
    @printf("load_tile: %.2f s, valid %.2f%%\n", time() - t0, 100 * sum(td.mask) / length(td.mask))

    t0 = time()
    r = compute_di(td; orientation=orientation)
    @printf("compute_di: %.2f s\n", time() - t0)

    @printf("DI raw range: %.6g .. %.6g\n", minimum(r.di[r.mask]), maximum(r.di[r.mask]))
    @printf("normalized min/max: %.6g / %.6g\n", r.lo, r.hi)

    mkpath(dirname(out_prefix))
    npzwrite(out_prefix * "_di.npz", Dict(
        "di_raw" => Float32.(r.raw),
        "di" => Float32.(r.di),
        "di_norm" => Float32.(r.di_norm),
        "mask" => UInt8.(r.mask),
    ))
    open(out_prefix * "_di_meta.json", "w") do io
        JSON3.write(io, Dict(
            "orientation" => orientation,
            "min" => r.lo,
            "max" => r.hi,
            "valid_frac" => sum(r.mask) / length(r.mask),
            "source" => prep_npz,
        ))
    end
    println("saved ", out_prefix, "_di.npz")
end

main(ARGS)
