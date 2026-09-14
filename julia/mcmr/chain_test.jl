# Does MCMRSimulator let spins move along a chain of overlapping spheres? Compare a
# 10 µm periodic chain (r = 0.5, centre spacing 0.25 µm) with an infinite cylinder r = 0.5,
# PGSE δ=6 Δ=12 along the chain axis (z) and perpendicular (x), D = 2, b = 1000.
using MCMRSimulator, MRIBuilder, StaticArrays, Statistics
L = 10.0; sc = MRIBuilder.Scanner(gradient=2000., slew_rate=1e6)
seq(g) = DWI(bval=1.0, diffusion_time=12., TE=22., scanner=sc, gradient=(type=:trapezoid, orientation=g, rise_time=0.1, δ=6.))
seqs = [seq([0.,0,1]), seq([1.,0,0])]
sig(x) = transverse(x) / length(x)
function run(geom, name; n=40000)
    sim = Simulation(seqs; geometry=geom, diffusivity=2.0)
    r = readout(n, sim; bounding_box=BoundingBox([0.,0,0],[L,L,L]), subset=[Subset(inside=true)])
    println(rpad(name, 34), " along z: ", round(sig(r[1,1]), digits=3), "  perp x: ", round(sig(r[2,1]), digits=3), "  n_in=", length(r[1,1]))
end
pos = [SVector(5.0, 5.0, z) for z in 0:0.25:L-0.25]
run(Spheres(position=pos, radius=fill(0.5, length(pos)), repeats=[L,L,L], overlapping=true),  "chain, overlapping=true")
run(Spheres(position=pos, radius=fill(0.5, length(pos)), repeats=[L,L,L], overlapping=false), "chain, overlapping=false")
run(Spheres(position=pos, radius=fill(0.5, length(pos)), repeats=[L,L,L], overlapping=true, permeability=Inf), "chain, overlapping=true, perm=Inf")
run(Cylinders(position=[SVector(5.0, 5.0)], radius=[0.5], repeats=[L,L]), "cylinder r=0.5 (reference)")
println("free along z would be exp(-bD) = ", round(exp(-2.0), digits=3), "; exit 0")
