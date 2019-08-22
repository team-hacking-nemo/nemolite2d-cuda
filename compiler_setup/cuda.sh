#!/bin/bash

command -v module >/dev/null 2>&1 && {
    module load gcc
    module load cuda
}

command -v nvcc >/dev/null 2>&1 || {
    echo "Unable to find 'nvcc' CUDA compiler."
    return 1
}

NVCC_PATH=$(which nvcc)
CUDA_BIN=$(dirname "$NVCC_PATH")
CUDA_HOME=$(dirname "$CUDA_BIN")

export CUDA_INC="$CUDA_HOME/include"
export CUDA_LIB="$CUDA_HOME/lib64"

export C_INCLUDE_PATH="$CUDA_INC:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="$CUDA_LIB:$LD_LIBRARY_PATH"

# Always using Volta GV100 (either Quadro or Tesla) that supports compute
# capacity 7.0.
export CUDA_ARCH=sm_70

echo "CUDA home directory: $CUDA_HOME"

export NVCC_FLAGS=" -O3 -std=c++14 -use_fast_math -arch=$CUDA_ARCH --ptxas-options=-v -lineinfo"
