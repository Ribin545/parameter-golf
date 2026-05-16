#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — One-command environment setup for parameter-golf
# Works on: Windows WSL, RunPod, Ubuntu 22.04+
# =============================================================================
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-}"
VENV_NAME="${VENV_NAME:-pg_env}"
VENV_DIR="$HOME/.venvs/$VENV_NAME"



# =============================================================================
echo "=========================================="
echo "  Parameter Golf Environment Setup"
echo "=========================================="
echo "Python: $PYTHON_VERSION"
echo "VENV: $VENV_DIR"
echo ""

# --- Pre-flight: NVIDIA Driver + CUDA Toolkit check ---
echo "[preflight] Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
    echo "[preflight] NVIDIA driver detected: $DRIVER_VER"
else
    echo "[ERROR] nvidia-smi not found. NVIDIA driver is required."
    echo "[ERROR] On Windows WSL: install the driver from https://www.nvidia.com/Download/index.aspx"
    echo "[ERROR] On Ubuntu: sudo apt install -y nvidia-driver-560  # or latest"
    exit 1
fi

echo "[preflight] Checking CUDA toolkit (nvcc)..."
if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release //; s/,.*//')
    echo "[preflight] CUDA toolkit detected: $NVCC_VER"
else
    echo "[preflight] nvcc not found. CUDA toolkit is needed for cu130 wheels."
    echo "[preflight] Attempting to install CUDA 13.0 toolkit..."
    if command -v apt-get &> /dev/null; then
        UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "ubuntu2404")
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_CODENAME}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb || \
            wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update -qq
        sudo apt-get install -y -qq cuda-toolkit-13-0
        if command -v nvcc &> /dev/null; then
            echo "[preflight] CUDA 13.0 toolkit installed successfully."
        else
            echo "[ERROR] CUDA toolkit installation failed."
            exit 1
        fi
    else
        echo "[ERROR] Cannot auto-install CUDA toolkit. Please install manually:"
        echo "    https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
fi


# --- Step 1: Ensure Python is installed ---
PYTHON_CMD=$(command -v python$PYTHON_VERSION 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
if [ -z "$PYTHON_CMD" ]; then
    echo "[install] Python not found. Installing python$PYTHON_VERSION..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3 python3-venv python3-dev python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip
    elif command -v conda &> /dev/null; then
        conda create -n $VENV_NAME python=$PYTHON_VERSION -y
    else
        echo "[ERROR] No package manager found. Please install Python manually."
        exit 1
    fi
    PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")
fi

echo "[check] Python command: $PYTHON_CMD"
$PYTHON_CMD --version

# Check Python version compatibility with PyTorch
PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
echo "[check] Detected Python ${PY_MAJOR}.${PY_MINOR}"

if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 14 ]; then
    echo "[WARNING] Python 3.14+ detected. PyTorch cu130 has limited wheels for 3.14."
    echo "[WARNING] If torch import fails, your system may need the CUDA 13.0 toolkit installed."
fi

# --- Step 2: Create virtual environment ---
echo "[setup] Creating virtual environment at $VENV_DIR..."
mkdir -p "$HOME/.venvs"
$PYTHON_CMD -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- Step 3: Upgrade pip ---
echo "[setup] Upgrading pip..."
python -m pip install --upgrade pip --quiet

# --- Step 4: Install PyTorch with CUDA support ---
echo "[setup] Installing PyTorch + CUDA..."
# Detect CUDA version and install matching PyTorch
CUDA_VER=$(python -c "import subprocess; out=subprocess.run(['nvcc','--version'], capture_output=True, text=True); print([l for l in out.stdout.split('\\n') if 'release' in l][0].split('release ')[1].split(',')[0])" 2>/dev/null || echo "12.4")

echo "[info] Detected CUDA version: $CUDA_VER"
case "$CUDA_VER" in
    13.*)
        echo "[install] CUDA 13.0 detected → PyTorch 2.11.0+cu130"
        echo "[WARNING] cu130 requires CUDA 13.0 system libraries (libcudart.so.13)."
        echo "[WARNING] If import torch fails with 'libcudart.so.13 not found', install CUDA 13.0 toolkit first:"
        echo "    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
        echo "    sudo dpkg -i cuda-keyring_1.1-1_all.deb"
        echo "    sudo apt-get update && sudo apt-get install -y cuda-toolkit-13-0"
        python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
        ;;
    12.8|12.6)
        echo "[install] CUDA 12.8/12.6 detected → PyTorch stable cu128"
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
        ;;
    12.4|12.1)
        echo "[install] CUDA 12.4/12.1 detected → PyTorch stable cu124"
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
        ;;
    11.8)
        echo "[install] CUDA 11.8 detected → PyTorch stable cu118"
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
        ;;
    *)
        echo "[install] Unknown CUDA $CUDA_VER → Installing CPU-only PyTorch (fallback)"
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# --- Step 5: Install remaining dependencies ---
echo "[setup] Installing remaining dependencies from requirements.txt..."
python -m pip install --quiet \
    numpy \
    tqdm \
    huggingface-hub \
    kernels \
    setuptools \
    typing-extensions==4.15.0 \
    datasets \
    tiktoken \
    sentencepiece \
    triton

# --- Step 6: Verify installation ---
echo ""
echo "=========================================="
echo "  Verification"
echo "=========================================="
python -c "
import torch, triton, sentencepiece, numpy, sys
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Triton:', triton.__version__)
print('NumPy:', numpy.__version__)
print('SentencePiece: OK')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Or on Windows: $VENV_DIR\\Scripts\\activate.bat"
echo ""