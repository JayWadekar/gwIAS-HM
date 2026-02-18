import os
import sys
from datetime import date

REPO_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".."))
PIPELINE_DIR = os.path.join(REPO_ROOT, "Pipeline")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PIPELINE_DIR)

project = "GWIAS-HM API"
author = "GWIAS-HM contributors"
copyright = f"{date.today().year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Keep builds robust when optional scientific deps are unavailable.
autodoc_mock_imports = [
    "Fr",
    "pylal",
    "lal",
    "lalsimulation",
    "gwosc",
    "requests",
    "cogwheel",
    "torch",
    "pytorch_lightning",
    "nflows",
    "sklearn",
    "pyfftw",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["api.css"]

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "show_nav_level": 1,
}
