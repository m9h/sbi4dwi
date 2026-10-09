"""
Pseudo-CT Module for Skull Conductivity Estimation.
Synergy between Transcranial Focused Ultrasound (TFUS) and EIT.
Implements mappings from MRI -> Pseudo-CT (HU) -> Porosity -> Conductivity.

References:
1. Plymouth mr-to-pct: https://github.com/sitiny/mr-to-pct
2. BabelBrain: https://github.com/BabelViscoFDTD/BabelBrain
3. Archie's Law for porous media conductivity.
"""

import jax.numpy as jnp

class PseudoCTMapper:
    """
    Maps MRI data to Electrical Conductivity via Pseudo-CT density estimation.
    """
    
    def __init__(self, method='plymouth'):
        """
        Args:
            method: 'plymouth' (T1-based) or 'babel' (UTE/ZTE-based).
        """
        self.method = method
        
    def mri_to_hu(self, mri_intensity):
        """
        Maps MRI intensity to Pseudo-CT Hounsfield Units (HU).
        """
        if self.method == 'plymouth':
            # Simplified Plymouth Logic (T1 -> HU)
            # Typically involves identifying bone mask and scaling.
            # Here we assume mri_intensity is normalized [0, 1] within the skull mask.
            # Bone HU range: ~300 to ~2000+. 
            # In T1, bone is dark (low signal). In UTE, bone has signal.
            
            # For T1: Inverse relationship? Darker = Denser? 
            # Or reliance on shape/atlas. Plymouth usually uses deep learning (UNet).
            # For this heuristic implementation, we'll assume an INVERSE mapping
            # for T1 (Low T1 = Cortical Bone = High Density).
            
            # Heuristic: HU = 2000 * (1 - intensity) + 300
            return 1700.0 * (1.0 - mri_intensity) + 300.0
            
        elif self.method == 'babel':
            # BabelBrain (UTE/ZTE) Logic
            # Bone signal correlates positively with density in UTE/ZTE/PETRA.
            # Heuristic: Linear map
            return 2000.0 * mri_intensity + 100.0
            
        else:
            return mri_intensity # Pass through

    def hu_to_porosity(self, hu):
        """
        Maps Hounsfield Units to Porosity (Phi).
        Bone Mineral Density (BMD) ~ HU.
        Porosity is inverse of density.
        
        Model:
        Phi = 1 - (HU / HU_max)
        Assuming HU_max (cortical bone) ~ 2000-3000.
        """
        HU_MAX = 2500.0
        # Clip
        hu_vals = jnp.clip(hu, 0, HU_MAX)
        phi = 1.0 - (hu_vals / HU_MAX)
        
        # Ensure minimal porosity (network of lacunae/canaliculi always exists)
        # Minimum porosity ~ 5%
        return jnp.maximum(phi, 0.05)

    def archies_law(self, porosity, sigma_brine=2.0, m=1.5):
        """
        Archie's Law for Conductivity.
        sigma_skull = sigma_brine * porosity^m
        
        Args:
            porosity: Phi (0 to 1).
            sigma_brine: Conductivity of fluid in pores (CSF/Saline ~ 1.5 - 2.0 S/m).
            m: Cementation exponent (typically 1.3 - 2.5). 1.5 is common for skull.
            
        Returns:
            Sigma (conductivity) in S/m.
        """
        return sigma_brine * (porosity ** m)

    def predict_conductivity(self, mri_intensity):
        """
        End-to-end pipeline: MRI -> HU -> Porosity -> Conductivity.
        """
        hu = self.mri_to_hu(mri_intensity)
        phi = self.hu_to_porosity(hu)
        sigma = self.archies_law(phi)
        return sigma
