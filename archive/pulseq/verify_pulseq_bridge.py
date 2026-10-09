
import sys
import os
import jax.numpy as jnp
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Mock pypulseq if not installed, so we can verify the bridge logic
try:
    import pypulseq
except ImportError:
    # Create a mock module structure
    sys.modules["pypulseq"] = MagicMock()
    
from dmipy_jax.external.pulseq import pulseq_to_general, check_pypulseq_installed

def verify_bridge():
    print("Verifying PyPulseq Bridge...")
    
    # Check import check
    try:
        check_pypulseq_installed()
        print("PyPulseq is installed (or mocked).")
    except ImportError:
        print("Pass: Correctly detected missing pypulseq.")
        return

    # Mock a sequence object
    mock_seq = MagicMock()
    # Mock duration
    mock_seq.duration.return_value = 10e-3 # 10ms
    # Mock blocks
    mock_block = MagicMock()
    mock_block.duration = 5e-3
    mock_block.gx = MagicMock() # Has gradients
    mock_seq.blocks = [1, 2] # Dummy pointers
    mock_seq.get_block.return_value = mock_block
    
    # Run conversion
    # Note: Our current implementation has placeholders, so we expect it to run without error 
    # but maybe produce zeros.
    print("Converting mock sequence...")
    general_seq = pulseq_to_general(mock_seq, dt=1e-3)
    
    print("Conversion successful.")
    print(f"Time points: {general_seq.time_points.shape}")
    print(f"Gradients: {general_seq.gradients.shape}")
    
    if general_seq.gradients.shape[0] == 10:
        print("Pass: Grid size matches duration (10ms / 1ms = 10 pts).")
    else:
        print(f"Fail: Expected 10 pts, got {general_seq.gradients.shape[0]}")

if __name__ == "__main__":
    verify_bridge()
