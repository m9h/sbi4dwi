"""
UCLH Stroke EIT Dataset Loader.
Source: https://github.com/EIT-ucl/Stroke-EIT-Dataset
Contains EIT voltages, MRI/CT references for Stroke patients.
"""

import os
import requests
import zipfile
import jax.numpy as jnp
from typing import Dict, Any


REPO_ZIP_URL = "https://github.com/EIT-team/Stroke_EIT_Dataset/archive/refs/heads/master.zip"
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dmipy_jax", "uclh_eit")

def download_uclh_dataset(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Downloads the UCLH Stroke EIT Dataset from GitHub.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "Stroke-EIT-Dataset.zip")
    extract_dir = os.path.join(data_dir, "extracted")
    
    # Check extraction
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        # Look for the main folder inside (Stroke_EIT_Dataset-master)
        for d in os.listdir(extract_dir):
            if "Stroke_EIT_Dataset" in d:
                return os.path.join(extract_dir, d)
        return extract_dir

    if not os.path.exists(zip_path):
        print("Downloading UCLH Stroke EIT Dataset from GitHub...")
        response = requests.get(REPO_ZIP_URL, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        
    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    
    # Return the inner folder
    for d in os.listdir(extract_dir):
        if "Stroke_EIT_Dataset" in d:
            return os.path.join(extract_dir, d)
    return extract_dir

def load_patient_data(patient_id: str = "01") -> Dict[str, Any]:
    """
    Loads data for a specific patient.
    """
    root = download_uclh_dataset()
    
    data_paths = {
        "root": root,
        "dataset_mat": os.path.join(root, "UCL_Stroke_EIT_Dataset.mat"),
        "json_data": os.path.join(root, "EITDATA.json"),
        "readme": os.path.join(root, "readme.md")
    }
    
    # Check if files exist
    if not os.path.exists(data_paths["dataset_mat"]):
        print(f"Warning: Main dataset file not found at {data_paths['dataset_mat']}")
        
    return data_paths

def list_patients():
    """Lists available subjects in the downloaded cache."""
    root = download_uclh_dataset()
    return os.listdir(root)
