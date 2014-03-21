#!/usr/bin/env bash

ln -s /getter_data data/getter_data
opencv_createsamples -vec out/cascade -info repo_data/info_markedup.dat