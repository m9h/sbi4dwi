# Along-chain diffusion in MCMRSimulator v1.1.0 vs sphere spacing (r = 0.5 µm, chain along z, 10 µm periodic).
# Pairwise-only overlaps (spacing ≥ 0.5) vs CATERPillar-like dense chains (spacing 0.25 → each sphere overlaps 3 on each side).
using MCMRSimulator, MRIBuilder, StaticArrays, Statistics
L = 10.0; sc = MRIBuilder.Scanner(gradient=2000., slew_rate=1e6)
seqz = DWI(bval=1.0, diffusion_time=12., TE=22., scanner=sc, gradient=(type=:trapezoid, orientation=[0.,0,1], rise_time=0.1, δ=6.))
sig(x) = transverse(x) / length(x)
for spacing in (0.9, 0.7, 0.5, 0.35, 0.25)
    pos = [SVector(5.0, 5.0, z) for z in 0:spacing:L-spacing/2]
    geom = Spheres(position=pos, radius=fill(0.5, length(pos)), repeats=[L,L,L], overlapping=true)
    sim = Simulation([seqz]; geometry=geom, diffusivity=2.0)
    r = readout(60000, sim; bounding_box=BoundingBox([0.,0,0],[L,L,L]), subset=[Subset(inside=true)])
    println("spacing ", spacing, " (", length(pos), " spheres): S along chain at b=1000 = ", round(sig(r[1,1]), digits=3), "  n_in=", length(r[1,1]))
end
println("cylinder reference ≈ 0.135 (free); exit 0")
