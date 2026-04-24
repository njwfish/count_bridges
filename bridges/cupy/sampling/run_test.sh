#!/bin/bash
export LD_LIBRARY_PATH=/home/njwfish/miniconda3/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/njwfish/miniconda3/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/njwfish/miniconda3/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_PATH=/home/njwfish/miniconda3

cd /orcd/data/omarabu/001/njwfish/cell-types/cell_types/generative/counting_flows/bridges/cupy/sampling
python test_bessel.py
