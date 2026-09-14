# MCMRSimulator.jl v1.1.0 parity run on a CATERPillar substrate (doc 007 §8.5).
# usage: julia +1.13 --project=julia/mcmr julia/mcmr/substrate_pgse.jl <tag> [nspins] [ndirs]
#   reads  validation/mcmr/<tag>_spheres.csv (x,y,z,r in µm) and <tag>_ref.csv (scheme + JAX MC reference)
#   writes validation/mcmr/<tag>_mcmr.csv  (per measurement: S_intra, S_extra from MCMRSimulator)
using MCMRSimulator, MRIBuilder, CSV, DataFrames, StaticArrays, JSON3, Statistics

tag    = length(ARGS) >= 1 ? ARGS[1] : "straight_0"
nspins = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20_000
ndirs  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0        # 0 = all measurements

meta = JSON3.read(read("validation/mcmr/meta.json", String))[Symbol(tag)]
L = Float64(meta.box_um); δ = Float64(meta.delta_ms); Δ = Float64(meta.Delta_ms); D = Float64(meta.D_um2_ms)

df  = CSV.read("validation/mcmr/$(tag)_spheres.csv", DataFrame)
pos = [SVector(r.x, r.y, r.z) for r in eachrow(df)]
geom = Spheres(position=pos, radius=Float64.(df.r), repeats=[L, L, L], overlapping=true, permeability=0.0)
println("geometry: $(length(pos)) spheres, box $L µm, δ=$δ Δ=$Δ D=$D")

ref = CSV.read("validation/mcmr/$(tag)_ref.csv", DataFrame)
idx = ndirs > 0 ? unique(vcat(1, round.(Int, range(2, nrow(ref), length=ndirs)))) : collect(1:nrow(ref))
TE = Δ + δ + 4.0
scanner = MRIBuilder.Scanner(gradient=2000.0, slew_rate=1e6)   # δ=6/Δ=12 at b=3000 needs ~340 mT/m
function pgse(b_smm2, g)
    bval = b_smm2 / 1000                                # ms/µm²
    g = b_smm2 < 50 ? (1.0, 0.0, 0.0) : g              # b0 row has a zero gradient vector
    DWI(bval=bval, diffusion_time=Δ, TE=TE, scanner=scanner,
        gradient=(type=:trapezoid, orientation=collect(g), rise_time=0.1, δ=δ))
end
seqs = [pgse(ref.b_s_mm2[i], (ref.gx[i], ref.gy[i], ref.gz[i])) for i in idx]
println("built $(length(seqs)) sequences (TE=$TE ms)")

sim = Simulation(seqs; geometry=geom, diffusivity=D)
bb  = BoundingBox([0.0, 0.0, 0.0], [L, L, L])
t0 = time()
res = readout(nspins, sim; bounding_box=bb, subset=[Subset(inside=true), Subset(inside=false)])
println("readout done in $(round(time()-t0, digits=1)) s; result size $(size(res))")
sig(x) = transverse(x) / length(x)
out = DataFrame(i=Int[], b=Float64[], gx=Float64[], gy=Float64[], gz=Float64[], S_intra_mcmr=Float64[], S_extra_mcmr=Float64[],
                S_intra_jax=Float64[], S_extra_jax=Float64[], n_in=Int[], n_out=Int[])
for (k, i) in enumerate(idx)
    r_in = ndims(res) == 2 ? res[k, 1] : res[k, end, 1]; r_out = ndims(res) == 2 ? res[k, 2] : res[k, end, 2]
    push!(out, (i, ref.b_s_mm2[i], ref.gx[i], ref.gy[i], ref.gz[i], sig(r_in), sig(r_out), ref.S_intra[i], ref.S_extra[i], length(r_in), length(r_out)))
end
# normalise by the b=0 measurement of each compartment
b0 = findfirst(out.b .< 50)
if b0 !== nothing
    out.S_intra_mcmr ./= out.S_intra_mcmr[b0]; out.S_extra_mcmr ./= out.S_extra_mcmr[b0]
end
CSV.write("validation/mcmr/$(tag)_mcmr.csv", out)
f_in = out.n_in[1] / (out.n_in[1] + out.n_out[1])
println("intra fraction of seeded spins: $(round(f_in, digits=3))  (JAX geometry f_intra $(meta.f_intra))")
for b in (1000.0, 2000.0, 3000.0)
    m = isapprox.(out.b, b; atol=50)
    any(m) || continue
    println("b=$b: S_intra mcmr $(round(mean(out.S_intra_mcmr[m]), digits=3)) jax $(round(mean(out.S_intra_jax[m]), digits=3)) | S_extra mcmr $(round(mean(out.S_extra_mcmr[m]), digits=3)) jax $(round(mean(out.S_extra_jax[m]), digits=3))  RMS diff intra $(round(sqrt(mean((out.S_intra_mcmr[m] .- out.S_intra_jax[m]).^2)), digits=3)) extra $(round(sqrt(mean((out.S_extra_mcmr[m] .- out.S_extra_jax[m]).^2)), digits=3))")
end
