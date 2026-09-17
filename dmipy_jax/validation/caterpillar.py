
import os
import subprocess
import tempfile
import shutil
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from dmipy_jax.simulation.sphere_sdf import MultiSphereSDF

class CATERPillarOracle:
    """
    Wrapper for the CATERPillar substrate generator.
    Handles configuration generation, execution, and parsing of sphere data.
    """
    def __init__(self, binary_path: str = "/home/mhough/dev/dmipy/vendor/CATERPillar/Caterpillar"):
        self.binary_path = binary_path
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"CATERPillar binary not found at {self.binary_path}. Please compile it.")

    def _write_config(self, config: Dict[str, Any], filepath: str):
        """Writes the configuration dictionary to a CATERPillar-compatible file."""
        with open(filepath, 'w') as f:
            for key, value in config.items():
                if key == "vox_sizes":
                    f.write("<vox_sizes>\n")
                    # Handle single value or list
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            f.write(f"{v}\n")
                    else:
                        f.write(f"{value}\n")
                    f.write("</vox_sizes>\n")
                else:
                    f.write(f"{key} {value}\n")
            f.write("<END>\n")

    def generate(self, config: Dict[str, Any], output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Runs CATERPillar with the given configuration.
        
        Args:
            config: Dictionary of parameters.
            output_dir: Directory to save outputs. If None, uses a temporary directory.
            
        Returns:
            pd.DataFrame: DataFrame containing sphere data (x, y, z, radius, type, id).
        """
        # Ensure output_dir exists or create temp
        temp_dir = False
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="caterpillar_run_")
            temp_dir = True
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        config_path = os.path.join(output_dir, "config.conf")
        
        # Enforce output directory in config
        config["data_directory"] = output_dir
        if "filename" not in config:
            config["filename"] = "simOutput"
            
        self._write_config(config, config_path)
        
        # Run binary
        cmd = [self.binary_path, config_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise RuntimeError(f"CATERPillar simulation failed: {e}")
            
        # Find output CSV
        # CATERPillar appends suffix like _rep00 or similar
        # Pattern: filename + suffix + "_spheres.csv"
        # We search for any csv ending in _spheres.csv in the dir
        csv_files = list(Path(output_dir).glob("*_spheres.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No sphere CSV output found in {output_dir}")
        
        # Load the latest one (or first)
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        
        # Clean up if temp
        if temp_dir:
            # We might want to keep it if debugging, but generally clean up
            # For now, let's keep it if something goes wrong? 
            # I'll just delete the config file but maybe we want to keep the CSV data in memory
            shutil.rmtree(output_dir)
            
        return df

    def get_sdf(self, df: pd.DataFrame) -> Callable[[Any], Any]:
        """
        Converts the sphere DataFrame into a JAX SDF function.
        """
        centers = df[['x', 'y', 'z']].values
        radii = df['radius'].values
        
        sdf_obj = MultiSphereSDF(centers, radii)
        return sdf_obj.get_sdf_func()

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Returns a default configuration for a quick test."""
        return {
            "repetitions": 1,
            "vox_sizes": [10],
            "data_directory": "/tmp",
            "filename": "test",
            "axons_without_myelin_icvf": 0.3,
            "axons_with_myelin_icvf": 0.0,
            "glial_pop1_icvf_soma": 0.05,
            "glial_pop1_icvf_branches": 0.1,
            "glial_pop2_icvf_soma": 0.0,
            "glial_pop2_icvf_branches": 0.0,
            "blood_vessels_icvf": 0.0,
            "spheres_overlap_factor": 4,
            "beading_variation": 0.0,
            "beading_variation_std": 0.0,
            "tortuous": 1,
            "alpha": 4,
            "beta": 0.25,
            "regrow_thr": 20,
            "min_rad": 0.2,
            "std_dev": 0.1,
            "ondulation_factor": 5,
            "beading_period": 10,
            "can_shrink": 1,
            "c2": 0.5,
            "nbr_threads": 1,
            "nbr_axons_populations": 1,
            "crossing_fibers_type": 0,
            "mean_glial_process_length": 15,
            "std_glial_process_length": 5
        }


class CATERPillarCLI:
    """CATERPillar ≥ 2026-09 (CMake build, `CATERPillar-cli --config x.json`, PascalCase JSON,
    space-delimited CSV with `cell_type cell_id component component_id parent_component_id
    X Y Z inner_radius outer_radius`). Returns the same DataFrame columns as `CATERPillarOracle`
    (x, y, z, radius, type, id) so `substrate_benchmark` can use either. No seed key exists
    upstream (std::random_device), so realisations are not reproducible from the config."""

    def __init__(self, binary_path: str = "/home/mhough/dev/_external/CATERPillar/build/CATERPillar-cli"):
        self.binary_path = binary_path
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"CATERPillar-cli not found at {self.binary_path}; build /home/mhough/dev/_external/CATERPillar with CMake.")

    @staticmethod
    def default_config(box_um=10.0, icvf=0.5, c2=0.98, tortuous=True, beading=0.0, beading_std=0.0, threads=8,
                       n_populations=1, crossing_type=0, min_radius=0.2, alpha=4.0, beta=0.25, epsilon=0.1):
        return {
            "GeneralParameters": {"OutputDirectory": "", "Filename": "substrate", "VoxelEdgeLength": box_um, "Repetitions": 1,
                                  "OverlappingFactor": 4, "NumberOfThreads": threads},
            "AxonParameters": {"AxonsICVF": 100.0 * icvf, "AxonsWithMyelinICVF": 0.0, "ArterioleICVF": 0.0, "CapillariesICVF": 0.0,
                               "CapillaryGamma": 3.0, "ArterioleRadiusMean": 6.0, "ArterioleRadiusStd": 1.0,
                               "NumberOfPopulations": n_populations, "CrossingFibersType": crossing_type,
                               "Alpha": alpha, "Beta": beta, "AlphaMyelin": 2.0, "BetaMyelin": 0.25, "MinRadius": min_radius,
                               "Tortuosity_Epsilon": epsilon, "FODF_c2": c2, "BeadingAmplitude": beading, "BeadingStd": beading_std,
                               "K1": 0.35, "K2": 0.006, "K3": 0.024, "Tortuous": bool(tortuous), "CanShrink": True,
                               "RegrowThreshold": 20, "UndulationFactor": 5},
            "GlialParameters": {**{f"Pop{i}SomaICVF": 0.0 for i in (1, 2, 3)}, **{f"Pop{i}ProcessesICVF": 0.0 for i in (1, 2, 3)},
                                **{f"Pop{i}SomaRadiusMean": 3.0 for i in (1, 2, 3)}, **{f"Pop{i}SomaRadiusStd": 0.5 for i in (1, 2, 3)},
                                **{f"Pop{i}MeanProcessLength": 30.0 for i in (1, 2, 3)}, **{f"Pop{i}StdProcessLength": 15.0 for i in (1, 2, 3)},
                                **{f"Pop{i}NbrPrimaryProcesses": 4 for i in (1, 2, 3)}, **{f"Pop{i}Branching": False for i in (1, 2, 3)}},
        }

    def generate(self, config: Dict[str, Any], output_dir: Optional[str] = None) -> pd.DataFrame:
        import json
        temp = output_dir is None
        output_dir = output_dir or tempfile.mkdtemp(prefix="caterpillar_cli_")
        os.makedirs(output_dir, exist_ok=True)
        config = json.loads(json.dumps(config)); config["GeneralParameters"]["OutputDirectory"] = output_dir
        cfg_path = os.path.join(output_dir, "config.json"); json.dump(config, open(cfg_path, "w"), indent=1)
        r = subprocess.run([self.binary_path, "--config", cfg_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"CATERPillar-cli failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        csvs = [p for p in Path(output_dir).glob("*.csv")]
        if not csvs:
            raise FileNotFoundError(f"no CSV in {output_dir}")
        raw = pd.read_csv(csvs[0], sep=r"\s+")
        ax = raw[raw["cell_type"] == "axon"]
        df = pd.DataFrame({"x": ax["X"].values, "y": ax["Y"].values, "z": ax["Z"].values,
                           "radius": ax["inner_radius"].values, "type": 0, "id": ax["cell_id"].values.astype(int),
                           "parent": ax["parent_component_id"].values.astype(int)})
        info = Path(output_dir).glob("*_growth_info.txt")
        df.attrs["growth_info"] = next((open(p).read() for p in info), "")
        if temp:
            shutil.rmtree(output_dir)
        return df
