# API Documentation

ViolaWake uses [pdoc](https://pdoc.dev/) for API reference generation. pdoc is zero-config and generates documentation directly from docstrings.

## Setup

Install the docs extra:

```bash
pip install -e ".[docs]"
```

## Generate HTML docs

```bash
pdoc violawake_sdk -o docs/api/html
```

This outputs static HTML to `docs/api/html/`. Open `docs/api/html/index.html` in a browser.

## Live preview (auto-reload)

```bash
pdoc violawake_sdk
```

This starts a local server (default http://localhost:8080) with live-reloading on source changes.

## Generate for a specific module

```bash
pdoc violawake_sdk.vad -o docs/api/html
pdoc violawake_sdk.wake_detector -o docs/api/html
```

## Notes

- `docs/api/html/` is git-ignored; generate locally or in CI
- pdoc reads `__docformat__` and supports Google, NumPy, and reStructuredText docstring styles
- For CI publishing, add a workflow step: `pdoc violawake_sdk -o public/api` and deploy the `public/` directory
