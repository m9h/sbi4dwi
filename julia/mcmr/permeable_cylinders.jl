# Ground truth for the exchange model (doc 008 §9.3): randomly packed permeable
# cylinders along z in MCMRSimulator v1.1.0, PGSE signals for two protocols
# (the DiSCo/PRISM single-Δ timing and the Fisher-optimised multi-Δ one),
# 32 directions per shell, all spins (exchange mixes compartments).
# usage: julia +1.13 -t 16 --project=julia/mcmr julia/mcmr/permeable_cylinders.jl <permeability> [nspins] [seed]
using MCMRSimulator, MRIBuilder, CSV, DataFrames, StaticArrays, Random, Statistics, LinearAlgebra

perm   = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.0
nspins = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20_000
seed   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
L = 40.0; D = 2.0                                       # µm box (x,y), µm²/ms
Random.seed!(seed)
pos, rad = random_positions_radii(SVector(L, L), 0.6, 2; mean=1.0, variance=0.2, min_radius=0.4, max_radius=2.5)
geom = Cylinders(position=pos, radius=rad, repeats=[L, L], permeability=perm)
f_in = sum(π .* rad .^ 2) / L^2
println("cylinders: $(length(rad)) (mean r $(round(mean(rad), digits=2)) µm), intra fraction $(round(f_in, digits=3)), permeability $perm")

rng = MersenneTwister(0); dirs = [normalize(SVector(randn(rng, 3)...)) for _ in 1:32]
protocols = Dict(
    "disco"     => (b=[1.0, 2.0, 3.0], Δ=[35.8, 35.8, 35.8], δ=17.7),
    "optimised" => (b=[1.4, 5.0, 5.0], Δ=[15.0, 15.0, 51.5], δ=10.0),
)
sc = MRIBuilder.Scanner(gradient=2000.0, slew_rate=1e6)
sig(x) = transverse(x) / length(x)
for (name, p) in protocols
    seqs = []; rows = []
    for k in 1:3, g in dirs
        TE = p.Δ[k] + p.δ + 4.0
        push!(seqs, DWI(bval=p.b[k], diffusion_time=p.Δ[k], TE=TE, scanner=sc,
                        gradient=(type=:trapezoid, orientation=collect(g), rise_time=0.1, δ=p.δ)))
        push!(rows, (shell=k, b=p.b[k], Delta=p.Δ[k], delta=p.δ, gx=g[1], gy=g[2], gz=g[3]))
    end
    sim = Simulation(seqs; geometry=geom, diffusivity=D)
    t0 = time()
    res = readout(nspins, sim; bounding_box=BoundingBox([0.0, 0.0, -L/2], [L, L, L/2]))
    println("$name: $(length(seqs)) sequences in $(round(time()-t0, digits=0)) s")
    S = [sig(ndims(res) == 1 ? res[i] : res[i, end]) for i in 1:length(seqs)]
    # T2-free signal; normalise by the b→0 limit: simulate one b=0.001 sequence per protocol would be cleaner,
    # but MCMR without relaxation gives |M_xy| = 1 at TE, so S is already the attenuation.
    df = DataFrame(rows); df.S = S
    CSV.write("validation/mcmr/perm_$(replace(string(perm), "." => "p"))_$(name).csv", df)
    for k in 1:3
        m = df.shell .== k
        println("  shell $k (b=$(p.b[k]), Δ=$(p.Δ[k])): mean S $(round(mean(df.S[m]), digits=4))")
    end
end
open("validation/mcmr/perm_$(replace(string(perm), "." => "p"))_meta.txt", "w") do io
    println(io, "f_intra=$f_in n_cyl=$(length(rad)) mean_r=$(mean(rad)) D=$D L=$L perm=$perm nspins=$nspins")
end
println("exit 0")
