# Direct measurement of the intra→extra residence time for the permeable-cylinder substrates
# of permeable_cylinders.jl: seed spins inside, no gradients, fraction still inside at t.
using MCMRSimulator, MRIBuilder, StaticArrays, Random, Statistics, LinearAlgebra
L = 40.0; D = 2.0
Random.seed!(0)
pos, rad = random_positions_radii(SVector(L, L), 0.6, 2; mean=1.0, variance=0.2, min_radius=0.4, max_radius=2.5)
sc = MRIBuilder.Scanner(gradient=2000.0, slew_rate=1e6)
times = [0.05, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
perms = length(ARGS) >= 1 ? parse.(Float64, ARGS) : [0.0, 0.3, 1.0, 3.0]
for perm in perms
    geom = Cylinders(position=pos, radius=rad, repeats=[L, L], permeability=perm)
    seqs = [SpinEcho(TE=t, scanner=sc) for t in times]
    sim = Simulation(seqs; geometry=geom, diffusivity=D)
    snaps = readout(8000, sim; bounding_box=BoundingBox([0.0, 0.0, -L/2], [L, L, L/2]), return_snapshot=true)
    getsnap(i) = ndims(snaps) == 1 ? snaps[i] : snaps[i, end]
    in0 = [isinside(geom, s.position) > 0 for s in getsnap(1)]
    fr = Float64[]
    for i in 1:length(times)
        ins = [isinside(geom, s.position) > 0 for s in getsnap(i)]
        push!(fr, mean(ins[in0]))          # P(inside at t | inside at t≈0)
    end
    println("perm=$perm: fraction still inside at t=$(times) ms: $(round.(fr, digits=3))")
end
println("exit 0")
