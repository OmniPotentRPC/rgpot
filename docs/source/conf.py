#!/usr/bin/env python3

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../subprojects/doxyrest/sphinx"))

# -- Project information -----------------------------------------------------
project = "rgpot"
copyright = "2025--present, rgpot developers"
author = "Rohit Goswami"
html_logo = "../../branding/logo/rgpot_notext.svg"

# -- General configuration ---------------------------------------------------
extensions = [
    "doxyrest",
    "cpplexer",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
    "sphinxcontrib_rust",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "eon": ("https://eondocs.org", None),
}

# -- sphinxcontrib-rust configuration ----------------------------------------
rust_crates = {
    "rgpot_core": os.path.abspath("../../rgpot-core/"),
}
rust_doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crates")
rust_rustdoc_fmt = "rst"
rust_generate_mode = "always"

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]

# Shibuya theme specific options
html_theme_options = {
    "github_url": "https://github.com/OmniPotentRPC/rgpot",
}
html_baseurl = "rgpot.rgoswami.me"
