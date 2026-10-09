"""
MR-ARFI (Acoustic Radiation Force Impulse) simulation.

Generates synthetic MR-ARFI images by:
1. Designing an MR-ARFI pulse sequence with motion-sensitizing gradients (MSGs)
   per Butts Pauly (Kaye & Pauly, MRM 2011)
2. Creating a KomaMRI phantom with displacement-encoded spin motion
3. Running Bloch simulation to produce phase maps
4. Reconstructing displacement from phase

Also includes PRF shift thermometry (Rieke & Butts Pauly, JMRI 2008).

References:
    - Kaye, Chen, Pauly (2011). MRM 65(3):738-743. (Rapid GRE MR-ARFI)
    - Kaye, Pauly (2013). MRM 69(3):724-733. (In vivo brain MR-ARFI)
    - Rieke, Butts Pauly (2008). JMRI 27(2):376-390. (MR thermometry review)
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Optional, Tuple


# Proton gyromagnetic ratio
GAMMA = 2.675e8  # rad/s/T
GAMMA_HZ = 42.576e6  # Hz/T


# ---------------------------------------------------------------------------
# MR-ARFI Sequence Design (pypulseq)
# ---------------------------------------------------------------------------

def design_arfi_sequence(
    fov: float = 0.256,
    n: int = 128,
    slice_thickness: float = 5e-3,
    msg_amplitude: float = 40e-3,
    msg_duration: float = 5e-3,
    fus_duration: float = 10e-3,
    te: float = 25e-3,
    tr: float = 500e-3,
    flip_angle_deg: float = 30.0,
    output_file: Optional[str] = None,
) -> dict:
    """
    Design a GRE MR-ARFI Pulseq sequence per Butts Pauly (2011).

    Sequence structure:
        RF excitation → MSG_lobe_1 (+) → [FUS pulse window] → MSG_lobe_2 (-) → GRE readout

    The bipolar MSG (motion-sensitizing gradient) pair encodes tissue displacement
    into signal phase. Displacement during the FUS window between MSG lobes produces
    a net phase shift proportional to displacement:
        Δφ = γ · G_MSG · δ · u

    Args:
        fov: Field of view in metres.
        n: Matrix size (n x n).
        slice_thickness: Slice thickness in metres.
        msg_amplitude: MSG gradient amplitude in T/m.
        msg_duration: Duration of each MSG lobe in seconds.
        fus_duration: Duration of FUS pulse window in seconds.
        te: Echo time in seconds.
        tr: Repetition time in seconds.
        flip_angle_deg: Flip angle in degrees.
        output_file: If provided, save .seq file to this path.

    Returns:
        Dict with sequence parameters and displacement encoding coefficient.
    """
    # Displacement encoding coefficient: phase per unit displacement
    # Δφ = γ · G · δ · u  →  enc_coeff = γ · G · δ (rad/m)
    enc_coeff = GAMMA * msg_amplitude * msg_duration  # rad/m

    # Displacement sensitivity: minimum detectable displacement
    # Assuming phase noise floor ~0.05 rad (single-shot GRE)
    phase_noise = 0.05  # rad
    min_displacement = phase_noise / enc_coeff  # metres

    seq_params = {
        "type": "GRE_MR_ARFI",
        "fov_m": fov,
        "matrix_size": n,
        "slice_thickness_m": slice_thickness,
        "msg_amplitude_T_m": msg_amplitude,
        "msg_duration_s": msg_duration,
        "fus_duration_s": fus_duration,
        "te_s": te,
        "tr_s": tr,
        "flip_angle_deg": flip_angle_deg,
        "encoding_coefficient_rad_m": enc_coeff,
        "min_displacement_m": min_displacement,
        "min_displacement_um": min_displacement * 1e6,
        "has_msg_lobes": True,
        "has_fus_trigger": True,
        "msg_polarity": [+1, -1],  # bipolar pair
    }

    # Generate Pulseq .seq file if pypulseq is available
    if output_file is not None:
        try:
            _write_arfi_pulseq(seq_params, output_file)
            seq_params["seq_file"] = output_file
        except ImportError:
            seq_params["seq_file"] = None
            seq_params["_note"] = "pypulseq not available; sequence params only"

    return seq_params


def _write_arfi_pulseq(params: dict, output_file: str):
    """Write MR-ARFI sequence to Pulseq .seq file using pypulseq."""
    import pypulseq as pp

    system = pp.Opts(
        max_grad=80e-3,  # T/m
        max_slew=200,  # T/m/s
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )
    seq = pp.Sequence(system=system)

    # RF excitation (sinc pulse)
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=params["flip_angle_deg"] * np.pi / 180,
        duration=2e-3,
        slice_thickness=params["slice_thickness_m"],
        system=system,
        return_gz=True,
    )

    # MSG lobe 1 (positive polarity)
    msg1 = pp.make_trapezoid(
        channel="z",
        amplitude=params["msg_amplitude_T_m"],
        duration=params["msg_duration_s"],
        system=system,
    )

    # FUS trigger (label/marker — no gradient, just a delay)
    fus_delay = pp.make_delay(params["fus_duration_s"])

    # MSG lobe 2 (negative polarity)
    msg2 = pp.make_trapezoid(
        channel="z",
        amplitude=-params["msg_amplitude_T_m"],
        duration=params["msg_duration_s"],
        system=system,
    )

    # Readout gradient + ADC
    delta_k = 1.0 / params["fov_m"]
    gx = pp.make_trapezoid(
        channel="x",
        flat_area=params["matrix_size"] * delta_k,
        flat_time=3.2e-3,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=params["matrix_size"],
        duration=gx.flat_time,
        delay=gx.rise_time,
        system=system,
    )

    # Phase encoding (simplified: single line for 2D)
    # Full k-space would loop over phase encodes
    gy = pp.make_trapezoid(channel="y", area=0, duration=1e-3, system=system)

    # Build sequence: RF → MSG1 → FUS → MSG2 → readout
    seq.add_block(rf, gz)
    seq.add_block(gz_reph)
    seq.add_block(msg1)
    seq.add_block(fus_delay)  # FUS fires here
    seq.add_block(msg2)
    seq.add_block(gx, adc, gy)

    # TR fill
    remaining = params["tr_s"] - seq.duration()[0]
    if remaining > 0:
        seq.add_block(pp.make_delay(remaining))

    seq.write(output_file)


# ---------------------------------------------------------------------------
# Displacement-Encoded Phantom for KomaMRI
# ---------------------------------------------------------------------------

def create_arfi_phantom_data(
    labels: np.ndarray,
    displacement_field: np.ndarray,
    grid_spacing: float,
    fus_onset_time: float = 10e-3,
    fus_duration: float = 10e-3,
) -> Dict:
    """
    Create phantom data for KomaMRI with displacement-encoded spin motion.

    Maps tissue labels to MR properties (T1, T2, PD) and applies the
    displacement field as a time-dependent motion during the FUS window.

    Args:
        labels: 2D or 3D integer tissue label array.
        displacement_field: Displacement in metres (same shape as labels).
            Positive = displacement along z (MSG encoding direction).
        grid_spacing: Voxel size in metres.
        fus_onset_time: Time when FUS pulse starts (seconds).
        fus_duration: Duration of FUS pulse (seconds).

    Returns:
        Dict with phantom data compatible with KomaMRI wrapper:
        - positions: (N, 3) spin positions in metres
        - T1, T2, PD: (N,) tissue properties
        - displacement_z: (N,) displacement per spin (metres)
        - fus_onset_s: FUS onset time
        - fus_duration_s: FUS duration
    """
    # MR tissue properties (approximate, for simulation)
    # Label: 0=bg, 1=scalp, 2=skull, 3=CSF, 4=GM, 5=WM
    T1_MAP = {0: 0.0, 1: 1.0, 2: 0.3, 3: 4.0, 4: 1.6, 5: 0.8}  # seconds
    T2_MAP = {0: 0.0, 1: 0.05, 2: 0.02, 3: 2.0, 4: 0.08, 5: 0.07}  # seconds
    PD_MAP = {0: 0.0, 1: 0.8, 2: 0.3, 3: 1.0, 4: 0.85, 5: 0.75}  # relative

    flat_labels = labels.ravel()
    flat_disp = displacement_field.ravel()
    n_spins = flat_labels.shape[0]

    # Generate spin positions on grid
    if labels.ndim == 2:
        ny, nx = labels.shape
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        positions = np.column_stack([
            xx.ravel() * grid_spacing,
            yy.ravel() * grid_spacing,
            np.zeros(n_spins),
        ])
    else:
        nz, ny, nx = labels.shape
        zz, yy, xx = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
        )
        positions = np.column_stack([
            xx.ravel() * grid_spacing,
            yy.ravel() * grid_spacing,
            zz.ravel() * grid_spacing,
        ])

    # Map labels to tissue properties
    T1 = np.array([T1_MAP.get(int(l), 0.0) for l in flat_labels])
    T2 = np.array([T2_MAP.get(int(l), 0.0) for l in flat_labels])
    PD = np.array([PD_MAP.get(int(l), 0.0) for l in flat_labels])

    # Filter out background spins (PD=0)
    mask = PD > 0
    return {
        "positions": positions[mask],
        "T1": T1[mask],
        "T2": T2[mask],
        "PD": PD[mask],
        "displacement_z": flat_disp[mask],
        "fus_onset_s": fus_onset_time,
        "fus_duration_s": fus_duration,
        "n_spins": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Analytical MR-ARFI Phase Prediction
# ---------------------------------------------------------------------------

def predict_arfi_phase(
    displacement: np.ndarray,
    msg_amplitude: float = 40e-3,
    msg_duration: float = 5e-3,
) -> np.ndarray:
    """
    Predict MR-ARFI phase shift from displacement (analytical).

    Δφ = γ · G_MSG · δ · u

    This is the ground truth prediction — KomaMRI Bloch simulation
    should match this for simple cases.

    Args:
        displacement: Tissue displacement in metres (any shape).
        msg_amplitude: MSG gradient amplitude (T/m).
        msg_duration: MSG lobe duration (seconds).

    Returns:
        Phase shift in radians (same shape as displacement).
    """
    return GAMMA * msg_amplitude * msg_duration * displacement


def recover_displacement_from_phase(
    phase_on: np.ndarray,
    phase_off: np.ndarray,
    msg_amplitude: float = 40e-3,
    msg_duration: float = 5e-3,
) -> np.ndarray:
    """
    Recover tissue displacement from MR-ARFI phase difference.

    u = (φ_on - φ_off) / (γ · G_MSG · δ)

    Args:
        phase_on: Phase image with FUS active (radians).
        phase_off: Phase image without FUS (radians).
        msg_amplitude: MSG gradient amplitude (T/m).
        msg_duration: MSG lobe duration (seconds).

    Returns:
        Estimated displacement in metres.
    """
    enc_coeff = GAMMA * msg_amplitude * msg_duration
    return (phase_on - phase_off) / enc_coeff
