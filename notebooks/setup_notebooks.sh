#!/bin/zsh

cd "$(dirname "${(%):-%x}}")"

git config filter.clean_ipynb.clean "$(pwd)/ipynb_drop_output.py"
git config filter.clean_ipynb.smudge cat
git config filter.clean_ipynb.required true

ln -sfn ../data -t .
ln -sfn ../expdata -t .
ln -sfn ../models -t .
ln -sfn ../plot3d -t .
ln -sfn ../utils.py -t .
