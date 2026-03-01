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
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
    "sphinxcontrib_rust",
    "sphinx_rustdoc_postprocess",
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

# -- sphinx-rustdoc-postprocess configuration --------------------------------
rustdoc_postprocess_toctree_target = "api/index.rst"
rustdoc_postprocess_toctree_rst = """
Rust API (``rgpot-core``)
-------------------------

.. toctree::
   :maxdepth: 2

   ../crates/rgpot_core/lib
"""

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/OmniPotentRPC/rgpot",
    "accent_color": "teal",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Saddle point finding on potential energy surfaces",
                },
                {
                    "title": "rgpycrumbs",
                    "url": "https://rgpycrumbs.rgoswami.me",
                    "summary": "Python bindings for rgpot",
                },
            ],
        },
    ],
}

html_context = {
    "source_type": "github",
    "source_user": "OmniPotentRPC",
    "source_repo": "rgpot",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_baseurl = "https://rgpot.rgoswami.me/"
