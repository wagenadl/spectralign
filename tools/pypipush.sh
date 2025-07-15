#!/bin/sh

python3 -m build

twine upload dist/spectralign-0.4.2-py3-none-any.whl
