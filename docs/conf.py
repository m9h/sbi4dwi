# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------
project = 'SBI4DWI'
copyright = '2026, Rutger Fick, Demian Wassermann, and Contributors'
author = 'Morgan Hough; dmipy lineage: Rutger Fick, Demian Wassermann'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',          # Markdown support
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
    'sphinx.ext.apidoc',    # generates reference/ from the package at build time (Sphinx >= 8.2)
]

apidoc_modules = [
    {
        'path': '../dmipy_jax',
        'destination': 'reference',
        'exclude_patterns': ['**/tests/**', '**/examples/**', '**/benchmarks/**', '**/experiments/**'],
        'separate_modules': False,
        'max_depth': 3,
    },
]

# -- Mock imports for ReadTheDocs ------------------------------------------------
# These packages contain compiled C/CUDA extensions or heavy native dependencies
# that are not available in the ReadTheDocs build environment. Mocking them
# allows autodoc to introspect the Python source without actually importing them.
autodoc_mock_imports = [
    # JAX ecosystem
    "jax",
    "jaxlib",
    "equinox",
    "diffrax",
    "optax",
    "distrax",
    "lineax",
    "optimistix",
    "jaxtyping",
    "jaxopt",
    "jax_md",
    "blackjax",
    "flowjax",
    "numpyro",
    "chex",
    "scico",
    "e3nn_jax",
    "ttax",
    "gpjax",
    "vbjax",
    # Deep learning
    "torch",
    "torchvision",
    "timm",
    # Numerical / scientific
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "h5py",
    "sympy",
    "cvxpy",
    "xarray",
    # Neuroimaging / dMRI
    "nibabel",
    "nilearn",
    "dipy",
    "dmipy",
    "pypulseq",
    "healpy",
    "simpleitk",
    # Mesh / geometry
    "meshio",
    "trimesh",
    # Image processing
    "cv2",
    "skimage",
    # Plotting
    "matplotlib",
    "corner",
    "seaborn",
    # Data access / cloud
    "boto3",
    "botocore",
    "datalad",
    # Misc
    "tqdm",
    "requests",
    "annotated_types",
    "beartype",
    "tabulate",
    "ipython",
    "IPython",
    "uguide",
]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- autodoc-typehints configuration -----------------------------------------
always_use_bars_union = True

bibtex_bibfiles = ['references.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    # working notes, prompts and drafts that are not part of the site
                    'agent_prompts', 'notes', 'openlifu', 'outreach', 'prompts', 'theory', 'theory.rst',
                    'tools', 'visuals', 'workshop', 'sphinxext', 'buckling_layer_research.md', 'remidi_education.md',
                    'decisions/00[1-5]*', 'decisions/007*', 'decisions/002*']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = "SBI4DWI Documentation"

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('..'))

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "tasklist",
]
