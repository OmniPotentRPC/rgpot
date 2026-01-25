#!/usr/bin/env python3

import os
import sys

# -- Project information -----------------------------------------------------
project = "rgpot"
copyright = "2025, rgpot developers"
author = "Rohit Goswami"
# html_logo = "../../branding/logo/pycrumbs_notext.svg"

# -- General configuration ---------------------------------------------------
extensions = [
    "breathe",
    "sphinx.ext.viewcode",  # Adds '[source]' links
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "eon": ("https://eondocs.org", None),
}

# Breathe configuration
breathe_projects = {"rgpot": "../build/xml"}
breathe_default_project = "rgpot"

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]

# Shibuya theme specific options
html_theme_options = {
    "github_url": "https://github.com/Theochemui/rgpot",
    # "nav_links": [
    #     {"title": "EON Tools", "url": "eon_tools"},
    # ],
}

autoapi_dirs = ["../../rgpot"]
html_baseurl = "rgpot.rgoswami.me"
