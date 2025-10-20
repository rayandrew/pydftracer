# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# Don't add project root to path - we want to use the installed package from site-packages
# If we add the project root, Python will try to import from source which may cause issues
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock imports for packages that may not be available during doc build
autodoc_mock_imports = []

# Try to import the package
try:
    import dftracer.python  # noqa: F401

    print("✓ dftracer.python package found and imported successfully.")
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: dftracer.python package not found: {e}")
    print("API documentation will have limited information.")
    print("To generate full API docs, install the package: pip install -e .")
    # Don't mock - let it fail to show what's missing
    # autodoc_mock_imports = ['dftracer', 'dftracer.python']

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "pydftracer"
copyright = "%Y, Hariharan Devarajan, Ray Andrew Sinurat"
author = "Hariharan Devarajan, Ray Andrew Sinurat"

# The version info for the project
# Try to get version from the package
try:
    from importlib.metadata import version

    release = version("pydftracer")
    version = ".".join(release.split(".")[:2])
except Exception:
    version = "0.1"
    release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "myst_parser",  # For Markdown support
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add mappings for intersphinx - link to main DFTracer docs and Python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "dftracer": ("https://dftracer.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Use type stubs (.pyi files) for documentation
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True
