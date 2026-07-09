"""Sphinx configuration for the Migdal documentation."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Migdal"
author = "Migdal developers"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "Migdal"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "text": "Migdal",
    },
    "show_toc_level": 2,
    "navigation_depth": 2,
}
html_sidebars = {
    "**": [],
}

myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath", "amsmath"]

# Sphinx 8 still names this setting mathjax3_config; it writes window.MathJax,
# which MathJax 4 uses for the STIX2 output-font choice below.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml-nofont.js"
mathjax3_config = {
    "output": {
        "font": "mathjax-stix2",
    },
}

# APS and JPSJ DOI targets redirect through publisher pages that return 403 to
# Sphinx linkcheck, even though the DOI URLs are the canonical citation links.
linkcheck_ignore = [
    r"https://doi\.org/10\.1103/PhysRev\.99\.1140",
    r"https://doi\.org/10\.1103/PhysRev\.59\.673",
    r"https://doi\.org/10\.1103/PhysRev\.139\.A796",
    r"https://doi\.org/10\.1103/PhysRevB\.96\.035147",
    r"https://doi\.org/10\.1103/PhysRevB\.28\.5100",
    r"https://doi\.org/10\.1103/PhysRevB\.42\.2416",
    r"https://doi\.org/10\.1143/JPSJ\.49\.1267",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Read the Docs should not need the heavy scientific runtime stack just to
# render the introductory pages and public driver signatures.
autodoc_mock_imports = [
    "h5py",
    "matplotlib",
    "mpi4py",
    "mpi4pyscf",
    "numpy",
    "pyscf",
    "scipy",
    "sparse_ir",
]
