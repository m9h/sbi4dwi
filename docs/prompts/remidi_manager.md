# Task: Manage ReMiDi Oracle

**Objective**: Utilize the [ReMiDi (Reconstruction of Microstructure using a Differentiable diffusion MRI simulator)](https://github.com/BioMedAI-UCSC/ReMiDi) project as a robust test oracle. This involves managing its Docker container to produce reference reconstructions.

**Infrastructure**: NVIDIA DGX / Spark Cluster.
**Target**: `remidi_oracle` interface.

## Instructions

1.  **Container Setup**:
    - ReMiDi typically provides a Dockerfile.
    - If not, create `docker/Dockerfile.remidi` wrapping their dependencies (TensorFlow/PyTorch, dipy, etc.).
    - Ensure it runs on the DGX (GPU support).

2.  **Data Mounting**:
    - Define a standard mounting point `/data/input` and `/data/output`.
    - Script a wrapper `scripts/run_remidi.sh` that:
        - Accepts a BIDS dataset path.
        - Mounts it to the Docker container.
        - Executes the ReMiDi reconstruction pipeline.

3.  **Oracle Comparison**:
    - Develop `dmipy_jax.benchmarks.remidi_comparator` to:
        - Load ReMiDi output maps (FA, MD, ODFs).
        - Load `dmipy-jax` output maps.
        - Compute Voxel-wise Error (RMSE, SSIM).
        - Generate specific "Disagreement Maps" to highlight where JAX differs from ReMiDi.

4.  **Automation**:
    - Integrate this into the CI/CD pipeline (if feasible) or the "Nightly Benchmark" suite on the DGX.

## Deliverables
- [ ] `docker/Dockerfile.remidi` (or build script).
- [ ] `scripts/run_remidi.sh`.
- [ ] Comparison Module (`remidi_comparator`).
