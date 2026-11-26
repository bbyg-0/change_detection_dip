"""K&I thresholding untuk DI satu tile."""

include("UAVSARChangeDetection.jl")
using .UAVSARChangeDetection
using NPZ, JSON3, Printf

function main(args)
    length(args) >= 3 || error("Usage: julia run_ki.jl <di.npz> <out_prefix> <pdf> [L] [cost] [min_bins]")
    di_npz = args[1]
    out_prefix = args[2]
    pdf = args[3]
    L = length(args) >= 4 ? parse(Int, args[4]) : 2048
    cost_mode = length(args) >= 5 ? args[5] : "likelihood"
    min_bins = length(args) >= 6 ? parse(Int, args[6]) : 5

    z = npzread(di_npz)
    di_norm = z["di_norm"]
    mask = BitMatrix(z["mask"] .!= 0)

    t0 = time()
    r = ki_threshold(di_norm, mask; pdf=pdf, L=L, cost_mode=cost_mode, min_bins=min_bins)
    dt = time() - t0
    n = sum(mask)
    nchg = sum(r.change)
    @printf("pdf=%s cost=%s L=%d min_bins=%d\n", pdf, cost_mode, L, min_bins)
    @printf("T* = %d / %d (norm %.6f), J = %.6f, waktu %.3f s\n",
            r.T, L, r.T / L, r.J, dt)
    @printf("valid=%d, changed=%d (%.3f%%)\n", n, nchg, 100 * nchg / n)

    mkpath(dirname(out_prefix))
    npzwrite(out_prefix * "_ki.npz", Dict(
        "T" => r.T,
        "Js" => Float32.(r.Js),
        "change" => UInt8.(r.change),
        "hist" => Float32.(r.hist),
        "levels" => Int32.(r.levels),
    ))
    open(out_prefix * "_ki_meta.json", "w") do io
        JSON3.write(io, Dict(
            "pdf" => pdf,
            "cost_mode" => cost_mode,
            "L" => L,
            "min_bins" => min_bins,
            "T" => r.T,
            "J" => r.J,
            "threshold_norm" => r.T / L,
            "valid" => n,
            "changed" => nchg,
            "changed_frac" => nchg / n,
            "params" => r.params,
            "source" => di_npz,
        ))
    end
    println("saved ", out_prefix, "_ki.npz")
end

main(ARGS)
