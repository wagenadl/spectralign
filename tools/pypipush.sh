#!/bin/sh

python3 -m build

twine upload dist/spectralign-0.2.0-py3-none-any.whl
