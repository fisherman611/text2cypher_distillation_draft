# 1. Check if CUDA_HOME is already set
echo $CUDA_HOME

# 2. Find where nvcc actually lives
which nvcc
find /usr /opt ~/miniconda3 -name nvcc 2>/dev/null

# 3. Check conda env toolkit
echo $CONDA_PREFIX
ls $CONDA_PREFIX/bin/nvcc 2>/dev/null && echo "nvcc found in conda env" || echo "not in conda env"

# 4. Check common system paths
ls /usr/local/cuda/bin/nvcc 2>/dev/null
ls -d /usr/local/cuda-* 2>/dev/null

# 5. Quick summary — run this one-liner
python3 -c "import torch; print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
