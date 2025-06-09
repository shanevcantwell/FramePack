#!/bin/bash

MIN_CUDA=121
MAX_CUDA=129

usage() {
  echo "Usage: $0 <cuda_version>" >&2 # Print usage to standard error
  echo "  <cuda_version>: An integer between $MIN_CUDA and $MAX_CUDA (inclusive)." >&2
  exit 1 # Exit with a non-zero status to indicate an error
}

# --- Argument Count Check ---
# Check if exactly one argument was provided.
if [ "$#" -ne 1 ]; then
  echo "Error: Incorrect number of arguments provided ($#)." >&2
  usage # Show usage instructions and exit
fi

# --- Argument Validation ---

# 1. First, check if the argument looks like an integer.
#    This prevents errors in the numerical comparison below if the input isn't numeric.
if ! [[ "$1" =~ ^[+-]?[0-9]+$ ]]; then
  echo "Error: Argument '$1' is not a valid integer." >&2
  usage # Call usage function and exit
fi

# 2. Now that we know it's an integer format, check if it's within the numerical range.
#    Using (( ... )) for arithmetic comparison.
if (($1 < MIN_CUDA )) || (( $1 > MAX_CUDA )); then
  echo "Error: Argument '$1' is outside the valid CUDA range [$MIN_CUDA-$MAX_CUDA]." >&2
  usage # Call usage function and exit
fi

# --- Main Execution ---
PIP_STRING="pip install --force-reinstall --break-system-packages torch torchvision torchcodec torchaudio xformers --index-url https://download.pytorch.org/whl/cu$1"
echo "Executing $PIP_STRING..."
$PIP_STRING

exit 0 # Exit successfully
