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
CUDA_DIR=$(dirname "$CUDA_BIN")

export C_INCLUDE_PATH="$CUDA_DIR/include/:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64/:$LD_LIBRARY_PATH"
