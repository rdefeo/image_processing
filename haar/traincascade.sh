#!/usr/bin/env bash

opencv_traincascade -data out/cascade/test.xml -vec out/heels.vec -bg data/bg.txt -npos 25 -nneg 2800 -nstages 20