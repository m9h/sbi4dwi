# Binary artifacts

Model checkpoints, baseline models and sample volumes are no longer tracked in
git (P1 item 6, 2026-10-09). They stay on disk in a development checkout and are
mirrored, with their original relative paths, under:

    /data/datasets/sbi4dwi-artifacts/

`*.eqx`, `*.eqx.meta.npz`, `*.pt`, `*.pth`, `*.bin`, `*.log`, `*.nii[.gz]` are
ignored by git. Result arrays under `validation/*.npz` and figures under
`validation/*.png` remain tracked because the decision docs cite them.

The vendored AMICO tree (`benchmarks/external/AMICO`, 44 `.bin` direction
tables) was deleted; `validation/validate_hcp_retest.py` uses the `dmri-amico`
wheel from the `baselines` extra instead.

Files moved out of the index (restore with `cp -r /data/datasets/sbi4dwi-artifacts/<path> <path>`):

- `benchmarks/data/edden/sub-01/ses-02/dwi/sub-01_ses-02_dwi.nii.gz`
- `examples/results/diff_iso.nii.gz`
- `examples/results/diff_par.nii.gz`
- `examples/results/pv_ball.nii.gz`
- `examples/results/pv_stick.nii.gz`
- `examples/sbi/dry_run_model.eqx`
- `examples/sbi/global_mdn.eqx`
- `examples/sbi/uguide_model/torch_embedder.pt`
- `examples/sbi/uguide_model/torch_nf.pt`
- `experiments/ste_dataset_integration/grad_magnitude.nii.gz`
- `models/fwe_oracle.eqx`
- `test_data/dwi.nii.gz`
- `validation/dipy_refine_slurm1781.log`
- `validation/disco_force_protocol.log`
- `validation/disco_force_protocol_10.log`
- `validation/disco_force_protocol_p50.log`
- `validation/disco_microstructure_slurm1772.log`
- `validation/emulator_inversion.log`
- `validation/exchange_mcmr.log`
- `validation/exchange_mcmr_fastexchange.log`
- `validation/exchange_protocol_design.log`
- `validation/external/force_disco_ndi.log`
- `validation/external/force_ext_slurm1755.log`
- `validation/external/sbi_dmri_cluster_slurm1763.log`
- `validation/external/sbi_dmri_de_prism_nfib2.pt`
- `validation/external/sbi_dmri_slurm1761.log`
- `validation/flow4x_slurm1776.log`
- `validation/flow_disco_k3.eqx`
- `validation/flow_disco_k3.eqx.meta.npz`
- `validation/flow_disco_k3_dyad.eqx`
- `validation/flow_disco_k3_dyad.eqx.meta.npz`
- `validation/flow_dyad4x_slurm1779.log`
- `validation/flow_dyad_slurm1778.log`
- `validation/flow_prism_k2_spline_256x6.eqx`
- `validation/flow_sbi_run1_labelswitch.log`
- `validation/flow_sbi_run2_clustered.log`
- `validation/hcp_retest/qc_snr.log`
- `validation/hcp_retest_smoke.log`
- `validation/hybrid_disco_dyad_slurm1780.log`
- `validation/hybrid_disco_slurm1775.log`
- `validation/hybrid_posterior_slurm1773.log`
- `validation/hyperparam_cv_disco_ext_slurm1787.log`
- `validation/hyperparam_cv_slurm1784.log`
- `validation/mc_library_generation.log`
- `validation/mcmr/mcds/jax_dt_convergence.log`
- `validation/mcmr/mcds/parity_images.log`
- `validation/mcmr/mcds/parity_noimages.log`
- `validation/mcmr/perm_sweep.log`
- `validation/prism_posterior_disco_slurm1742.log`
- `validation/prism_substrate_b3_icvf07_slurm1788.log`
- `validation/prism_substrate_b3_slurm1786.log`
- `validation/prism_substrate_cli_slurm1785.log`
- `validation/prism_substrate_noiso_slurm1777.log`
- `validation/prism_substrate_results_slurm1739.log`
- `validation/prism_substrate_results_slurm1740.log`
- `validation/prism_substrate_results_slurm1744.log`
- `validation/runtime_hardi_slurm1782.log`
