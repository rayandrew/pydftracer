# pydftracer Documentation

This directory contains the Sphinx documentation for pydftracer.

Documentation is automatically built and hosted on [Read the Docs](https://readthedocs.org/).

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install manually:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

### Build HTML Documentation

```bash
cd docs
make html
```

The HTML documentation will be generated in `build/html/`. Open `build/html/index.html` in your browser.

### Clean Build Files

```bash
make clean
```

### Other Formats

```bash
make latexpdf  # Build PDF documentation (requires LaTeX)
make epub      # Build EPUB documentation
make help      # Show all available targets
```

## Documentation Structure

- `source/conf.py` - Sphinx configuration
- `source/index.rst` - Main documentation index
- `source/installation.rst` - Installation guide
- `source/quickstart.rst` - Quick start guide
- `source/ai_ml_guide.rst` - AI/ML tracing guide
- `source/dynamo_guide.rst` - PyTorch Dynamo integration guide
- `source/api/` - API reference documentation
  - `core.rst` - Core tracing API
  - `ai.rst` - AI/ML API
  - `dynamo.rst` - Dynamo API
  - `env.rst` - Environment configuration

## Contributing

When adding new modules or features:

1. Update the relevant `.rst` files in `source/api/`
2. Add examples to the appropriate guide
3. Rebuild and review the documentation
4. Check for broken links and formatting issues
