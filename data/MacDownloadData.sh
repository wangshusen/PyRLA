#!/usr/bin/env bash

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2" -o "YearPredictionMSD.bz2"
bzip2 -d YearPredictionMSD.bz2

