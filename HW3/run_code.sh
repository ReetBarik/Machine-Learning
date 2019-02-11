#!/bin/bash
git clone https://github.com/fmfn/BayesianOptimization.git
mv BayesianOptimization/* .
python setup.py install
python Q6.py > output.txt
python Q7.py >> output.txt
cat output.txt