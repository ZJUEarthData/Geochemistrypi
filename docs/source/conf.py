# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import sphinx_rtd_theme

project = "Geochemistry π"
copyright = "2023, ZJUEarthData"
author = "ZJUEarthData"
# release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = ["sphinxcontrib.napoleon", "sphinxcontrib.apidoc", "sphinx.ext.viewcode", "sphinx.ext.todo", "m2r2"]

templates_path = ["_templates"]
exclude_patterns = []

language = "EN"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "furo"
# html_theme = 'classic'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]

html_theme_options = {
    "top_of_page_button": "edit",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "announcement": "<em>More tutorial videos can be found on bilibili: ZJU_Earth_Data</em>",
    "light_css_variables": {
        "color-brand-primary": "blue",
        # "color-brand-content": "brown",
        "color-admonition-background": "orange",
        "font-stack": "Times New Roman, Times, serif",
    },
}


html_logo = "https://user-images.githubusercontent.com/66779478/239791119-3b1fe8c9-5f99-49f5-aa31-edf5a683372b.png"

source_suffix = [".rst", ".md"]


project_path = "../../geochemistrypi"
# autodoc_mock_imports = ["geochemistrypi"]
sys.path.insert(0, os.path.abspath("../.."))
# sys.path.insert(0, os.path.abspath(project_path))
print(os.path.abspath(project_path))
sys.path.insert(1, os.path.abspath("../../geochemistrypi/"))
sys.path.insert(2, os.path.abspath("../../geochemistrypi/data_mining/"))
sys.path.insert(3, os.path.abspath(".."))
sys.path.append("../geochemistrypy/geochemistrypy")
# ...

apidoc_module_dir = project_path
apidoc_output_dir = "python_apis"

apidoc_separate_modules = False

# Disable displaying the module name as a prefix to class and function names
add_module_names = False

# Disable displaying the module name as a prefix to class and function names
# autodoc_default_flags = ['package', 'undoc-members', 'private-members', 'special-members', 'inherited-members', 'show-inheritance', 'noindex']
html_show_sourcelink = False
html_use_smartypants = True
html_split_index = False
