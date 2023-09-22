#!/bin/bash
make clean
sphinx-apidoc -f -o . ../micropolarray
sphinx-build -M html . ./_build
#make html
sphinx-build -M latexpdf . ./_build
#make latexpdf