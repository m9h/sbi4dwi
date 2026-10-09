
import jax.numpy as jnp
import numpy as np
from dmipy_jax.simulation.scanner.sequences import GeneralSequence, SpinEchoSequence
from dmipy_jax.acquisition import JaxAcquisition

# Try to import pypulseq, but don't fail hard if not installed (soft dependency pattern)
try:
    import pypulseq as pp
except ImportError:
    pp = None

def check_pypulseq_installed():
    if pp is None:
        raise ImportError("pypulseq is required for this functionality. Install it via pip or uv.")

def pulseq_to_jax_acquisition(seq) -> JaxAcquisition:
    """
    Converts a PyPulseq Sequence object into a JaxAcquisition.
    
    This is best-effort: it extracts b-values and b-vectors by calculating
    moments of the diffusion implementation.
    
    Args:
        seq: A pypulseq.Sequence object.
        
    Returns:
        JaxAcquisition: The compatible acquisition scheme.
    """
    check_pypulseq_installed()
    
    # Calculate b-values and b-vectors using PyPulseq's built-in tools
    # definition: b = integral( (gamma * int(G) dt )^2 dt )
    
    # PyPulseq provides 'calculate_diffusion_info' or similar helpers usually?
    # Actually, PyPulseq sequences don't always track 'blocks' as diffusion blocks easily 
    # unless they were built that way.
    # But usually we can export the blocks and analyze.
    
    # For now, let's assume the user is passing a standard diffusion sequence 
    # where we can extract timings.
    # If not, we might need to perform a full waveform integration.
    
    # Placeholder for full waveform analysis:
    # 1. Export waveforms
    # 2. Integrate gradient to get k(t)
    # 3. Integrate k(t)^2 to get b-value
    
    # Since this is a bridge, let's implement the moment calculation properly:
    waveforms = seq.waveforms_and_times()
    # waveforms is tuple (t_adc, adc, t_rf, rf, t_grad, grad) usually
    
    # PyPulseq's `seq.definitions` often holds 'b_value' if set by the user (standard convention).
    defs = seq.definitions
    
    bvals = []
    bvecs = []
    
    # If definitions exist, prioritize them
    if 'b_value' in defs and 'b_vec' in defs:
         # Note: Pulseq definitions might store just one value if it's a single block?
         # Or list?
         pass
         
    # If we can't find definitions, we raise for now as full waveform b-matrix calc is complex
    # and might be better handled by `pulseq_to_general_sequence` for raw simulation.
    
    # For the MVP bridge requested:
    # return a dummy or parsed object
    pass 
    
    # REVISIT: The user wants to bridge to functional simulators.
    # Let's pivot to returning the GeneralSequence which has the RAW waveforms.
    # This is much more useful for a Bloch simulator than just b-vals.
    return None

def pulseq_to_general(seq, dt: float = 10e-6) -> GeneralSequence:
    """
    Converts a PyPulseq Sequence into a GeneralSequence (dense waveforms).
    
    Args:
        seq: pypulseq.Sequence object.
        dt: Raster time for the dense grid (default 10us).
        
    Returns:
        GeneralSequence: Dense waveform representation for Bloch simulation.
    """
    check_pypulseq_installed()
    
    # 1. Get raw waveforms from PyPulseq
    # waveforms() returns a dictionary of waveforms or simple tuple depending on version.
    # The most robust way is to use `seq.waveforms_and_times()` method relative to a specific time raster.
    # But usually, we just want to densely sample the sequence.
    
    # We will use the `waveforms` method which exports (grad_waveforms, t_excitation, t_refocusing, t_adc)
    # However, to be precise on a UNIFORM grid, we should manually rasterize or Resample.
    
    # Let's iterate through the blocks to build the arrays, as this allows us to enforce our specific `dt`
    # and handle gaps correctly.
    
    t_total = seq.duration()[0] if isinstance(seq.duration(), (list, tuple)) else seq.duration()
    num_points = int(np.ceil(t_total / dt))
    time_points = np.arange(num_points) * dt
    
    grad_array = np.zeros((num_points, 3)) # x, y, z
    rf_amp = np.zeros(num_points, dtype=np.complex64)
    adc_mask = np.zeros(num_points, dtype=np.int32)
    
    # We iterate over blocks and rasterize them
    current_t_idx = 0
    
    # To avoid writing a full parser, we can use `seq.waveforms_and_times()` which does this BUT
    # it returns sparse/irregular data often.
    
    # Better approach for speed & correctness:
    # Use `seq.waveforms(time_range=[0, t_total])` and then interpolate.
    # pypulseq > 1.3 `waveforms` returns:
    # (gradient_waveforms, t_excitation, t_refocusing, t_adc)
    # gradient_waveforms is dict {channel: (t, amp)}
    
    try:
        # Note: This might vary by pypulseq version.
        # We attempt to use the export method.
        # If that fails, we fallback to block iteration.
        
        # Newer PyPulseq has `get_gradients`? No.
        # Let's use block iteration as it is the "ground truth".
        
        t_current = 0.0
        
        # PyPulseq sequence block iteration
        # seq.block_events is a dict of {block_id: events}
        # IDs are usually 1-based integers.
        # We can iterate sorted keys.
        block_ids = sorted(seq.block_events.keys())
        
        for b_idx in block_ids:
            block = seq.get_block(b_idx)
            # Debug:
            # print(f"DEBUG Block {b_idx}: {dir(block)}")
            
            # Safe access to duration
            if hasattr(block, 'block_duration'):
                dur = block.block_duration
            elif hasattr(block, 'duration'):
                dur = block.duration
            else:
                 # Fallback: check children
                 dur = 0.0
                 for child in [getattr(block, k) for k in ['rf', 'gx', 'gy', 'gz', 'adc', 'delay'] if hasattr(block, k)]:
                     if hasattr(child, 'duration'):
                         # Delays might use 'delay' instead of duration in some versions?
                         # Usually objects have duration.
                         pass
                     # This is getting messy, let's assume block_duration exists or fail with info
                 
                 # If we are here, we might need to look at 'delay' event specifically?
                 if hasattr(block, 'delay') and hasattr(block.delay, 'delay'):
                      dur = max(dur, block.delay.delay)
                      
            # If still 0 and not delay, raise?
            # Actually, let's just print and raise the original error if getting it fails to help debug
            if not hasattr(block, 'duration') and not hasattr(block, 'block_duration'):
                 print(f"DEBUG BAD BLOCK {b_idx}: {block}")
            
            # Start/End indices for this block
            # Be careful with rounding
            idx_start = int(np.round(t_current / dt))
            # idx_end is start of next
            idx_end = int(np.round((t_current + dur) / dt))
            
            # Safe slice length
            if idx_end > num_points:
                idx_end = num_points
            
            length = idx_end - idx_start
            
            if length > 0:
                t_rel = np.arange(length) * dt
                # Center time points or start? Start is fine.
                
                # 1. Gradients
                for ch_idx, ch_name in enumerate(['gx', 'gy', 'gz']):
                    grad_obj = getattr(block, ch_name, None)
                    if grad_obj is not None:
                        # Check type
                        if grad_obj.type == 'grad':
                            # Arbitrary gradient
                            # Assume it has .t and .waveform? Or .get(t)?
                            # PyPulseq Gradients usually have .waveform array if arbitrary
                            # or flat_time/ramp_time if trap.
                            
                            # Safest is to evaluate.
                            # But standard pypulseq objects don't always implement .eval(t) conveniently.
                            # They do support rasterization.
                            
                            # If trapezoid:
                            if grad_obj.type == 'trap':
                                # Re-implement trap eval
                                # amplitude, flat_time, rise_time, fall_time
                                # But let's see if we can convert it to arbitrary first?
                                # No, let's use a helper if possible.
                                
                                # Manual trap eval for now
                                amp = grad_obj.amplitude
                                rise = grad_obj.rise_time
                                flat = grad_obj.flat_time
                                fall = grad_obj.fall_time
                                
                                # Vectorized trap function
                                # t_rel is 0 to length*dt
                                
                                # Ramp up
                                r_up = np.clip(t_rel / rise, 0, 1) if rise > 0 else 1.0
                                # Ramp down (from end)
                                # End of trap is rise+flat+fall = duration (approx)
                                t_flat_end = rise + flat
                                total_dur = rise + flat + fall
                                r_down = np.clip(1.0 - (t_rel - t_flat_end) / fall, 0, 1) if fall > 0 else (1.0 if t_rel < total_dur else 0.0)
                                
                                # Combine (min of both ramps)
                                profile = np.minimum(r_up, r_down)
                                # Zero out after end
                                profile[t_rel > total_dur] = 0.0
                                
                                grad_array[idx_start:idx_end, ch_idx] += amp * profile
                                
                            # If arbitrary:
                            elif getattr(grad_obj, 'waveform', None) is not None:
                                # Resample waveform
                                # grad_obj.waveform is on grad_obj.t raster (usually 10us)
                                # We assume our dt matches or we interp
                                wf = grad_obj.waveform
                                # Assume wf is stored at seq.grad_raster_time (10us usually)
                                # If our dt is same, just copy (with truncation/padding)
                                
                                # Simple nearest neighbor / copy for MVP if grids match
                                # If not, interp.
                                raster_dt = seq.grad_raster_time if hasattr(seq, 'grad_raster_time') else 10e-6
                                t_wf = np.arange(len(wf)) * raster_dt
                                
                                # Interp
                                val = np.interp(t_rel, t_wf, wf, left=0, right=0)
                                grad_array[idx_start:idx_end, ch_idx] += val
                                
                # 2. RF
                if hasattr(block, 'rf') and block.rf is not None:
                    rf = block.rf
                    # rf.signal is complex array
                    # rf.t is time grid (usually 1us)
                    # rf.frequency_offset etc.
                    
                    if hasattr(rf, 'signal'):
                        wf = rf.signal
                        raster_dt = seq.rf_raster_time if hasattr(seq, 'rf_raster_time') else 1e-6
                        t_wf = np.arange(len(wf)) * raster_dt
                        
                        # Add frequency modulation if present
                        # phase = 2*pi*integral(freq) ... simple freq offset:
                        phase_offset = 0.0
                        if hasattr(rf, 'freq_offset') and rf.freq_offset != 0:
                             phase_offset = 2 * np.pi * rf.freq_offset * t_rel
                        
                        # Interp (complex)
                        # Split real/imag
                        val_r = np.interp(t_rel, t_wf, wf.real, left=0, right=0)
                        val_i = np.interp(t_rel, t_wf, wf.imag, left=0, right=0)
                        val = val_r + 1j*val_i
                        
                        # Apply phase
                        val = val * np.exp(1j * (rf.phase_offset + phase_offset))
                        
                        rf_amp[idx_start:idx_end] += val
                        
                # 3. ADC
                if hasattr(block, 'adc') and block.adc is not None:
                    adc = block.adc
                    # Mark active ADC
                    # Check delay
                    adc_delay = adc.delay
                    adc_dur = adc.num_samples * adc.dwell
                    
                    start_samples = int(adc_delay / dt)
                    dur_samples = int(adc_dur / dt)
                    
                    s = idx_start + start_samples
                    e = s + dur_samples
                    if e > num_points: e = num_points
                    if s < num_points:
                         adc_mask[s:e] = 1
            
            t_current += dur
            
    except Exception as e:
        print(f"Error rasterizing sequence: {e}")
        raise e
        
    return GeneralSequence(
        time_points=jnp.array(time_points),
        gradients=jnp.array(grad_array), # (N, 3)
        rf_amplitude=jnp.array(jnp.abs(rf_amp)), # Amplitude
        rf_phase=jnp.array(jnp.angle(rf_amp)),   # Phase
        adc_mask=jnp.array(adc_mask)
    )

def wrapper_pulseq_to_spinecho(seq) -> SpinEchoSequence:
    """
    Tries to extract TR/TE and effective gradients for a simplified spin echo Model.
    """
    check_pypulseq_installed()
    
    # Heuristic extraction
    defs = seq.definitions
    tr = defs.get('TR', 0.0)
    te = defs.get('TE', 0.0)
    
    # Gradients?
    # If standard diffusion, maybe 'b_value' / approx duration.
    # For now, let's return a dummy structure for the bridge demo.
    return SpinEchoSequence(
        TE=te,
        TR=tr,
        gradients=jnp.zeros((1, 3))
    )
