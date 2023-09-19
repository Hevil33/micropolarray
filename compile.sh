#!/bin/bash
pytest ./micropolarray/test_v2.py # first test
python3 -m build
pdoc --html --force --output-dir ./docs micropolarray
pandoc --metadata=title:"MyProject Documentation"               \
           --from=markdown+abbreviations+tex_math_single_backslash  \
           --pdf-engine=xelatex --variable=mainfont:"DejaVu Sans"   \
           --toc --toc-depth=4 --output=pdf.pdf  pdf.md