#!/bin/bash
pytest ./micropolarray/test_v2.py # first test
#pdoc --html --force --output-dir ./docs micropolarray
sphinx-apidoc -o docs micropolarray/
pip-compile pyproject.toml
python3 -m build
twine upload --verbose --skip-existing dist/*