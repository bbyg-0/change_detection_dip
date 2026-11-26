module UAVSARChangeDetection

include("prep.jl")
include("omnibus.jl")
include("ki.jl")
include("baselines.jl")

export TileData, load_tile, compute_di, omnibus_raw, build_covariance
export ki_threshold, evaluate_T
export logratio_di, cfar_map, significance_map, total_power

end
