#!/bin/bash
pytest -v ./ # first test
#pdoc --html --force --output-dir ./docs micropolarray
cd docs/
./create_documentation.sh
cd ..
#sphinx-apidoc -o docs micropolarray/
echo "Ended documentation, starting compiling..."
pip-compile pyproject.toml --resolver=backtracking
echo "Ended compiling, starting build..."
python3.12 -m build
echo "Ended build, starting upload..."
twine upload --verbose --skip-existing dist/*