"""Jalankan baseline change detection untuk satu tile."""

include("UAVSARChangeDetection.jl")
using .UAVSARChangeDetection
using NPZ, JSON3, Printf

function main(args)
    length(args) >= 4 || error("Usage: julia run_baselines.jl <prep.npz> <meta.json> <di.npz> <out_prefix>")
    prep_npz, meta_json, di_npz, out_prefix = args[1], args[2], args[3], args[4]

    td = load_tile(prep_npz, meta_json)
    z = npzread(di_npz)
    di_norm = z["di_norm"]
    mask = BitMatrix(z["mask"] .!= 0)

    mkpath(dirname(out_prefix))
    results = Dict{String,Any}()

    # 1) log-ratio + K&I Gaussian
    lr = logratio_di(td)
    r1 = ki_threshold(lr.di_norm, mask; pdf="gaussian", L=2048,
                      cost_mode="likelihood", min_bins=5)
    results["logratio_ki_gaussian"] = Dict(
        "T" => r1.T, "J" => r1.J, "threshold_norm" => r1.T / 2048,
        "changed" => sum(r1.change), "changed_frac" => sum(r1.change) / sum(mask))
    npzwrite(out_prefix * "_logratio_ki_gaussian.npz", Dict(
        "change" => UInt8.(r1.change), "di_norm" => Float32.(lr.di_norm),
        "T" => r1.T, "Js" => Float32.(r1.Js)))
    @printf("logratio+KI(gauss): T=%d changed=%.3f%%\n", r1.T, 100 * sum(r1.change) / sum(mask))

    # 2) CFAR (gamma clutter, pfa 1%)
    r2 = cfar_map(di_norm, mask; pfa=1e-2)
    results["cfar_pfa1e-2"] = Dict(
        "threshold" => r2.threshold, "changed" => sum(r2.change),
        "changed_frac" => sum(r2.change) / sum(mask))
    npzwrite(out_prefix * "_cfar_pfa1e-2.npz", Dict("change" => UInt8.(r2.change),
                                                    "threshold" => r2.threshold))
    @printf("CFAR pfa=1e-2     : thr=%.6f changed=%.3f%%\n", r2.threshold,
            100 * sum(r2.change) / sum(mask))

    # 3) Signifikansi 5% dan 1%
    for alpha in (0.05, 0.01)
        r3 = significance_map(di_norm, mask; alpha=alpha)
        key = "signif_" * string(alpha)
        results[key] = Dict("threshold" => r3.threshold, "changed" => sum(r3.change),
                            "changed_frac" => sum(r3.change) / sum(mask))
        npzwrite(out_prefix * "_" * key * ".npz", Dict("change" => UInt8.(r3.change),
                                                       "threshold" => r3.threshold))
        @printf("signif %.2f        : thr=%.6f changed=%.3f%%\n", alpha, r3.threshold,
                100 * sum(r3.change) / sum(mask))
    end

    open(out_prefix * "_baselines.json", "w") do io
        JSON3.write(io, results)
    end
    println("saved ", out_prefix, "_baselines.json")
end

main(ARGS)
