# Sphinx API Docs

This folder contains a standalone Sphinx site for code documentation and API reference.

## Quick start

```bash
cd /Users/tejaswi/Work/gwIAS-HM/docs/sphinx
python -m pip install -r requirements.txt
make html
```

Build output:
- `/Users/tejaswi/Work/gwIAS-HM/docs/sphinx/build/html/index.html`

## Notes
- The API pages are driven by docstrings from modules in `/Users/tejaswi/Work/gwIAS-HM/Pipeline`.
- Some heavy scientific dependencies are mocked in `source/conf.py` so docs can build in lightweight environments.
