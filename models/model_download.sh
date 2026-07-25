#!/bin/bash
set -e

PY_BIN=$(command -v python3)
if [ -z "$PY_BIN" ]; then
    echo "Hata: python3 bulunamadı." >&2
    exit 1
fi

PY_VERSION=$($PY_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_VENV_PKG="python${PY_VERSION}-venv"

echo "Gerekli kurulumlar yapılıyor..."
sudo apt update -qq
sudo apt install -y "${PY_VENV_PKG}"

"$PY_BIN" -m venv model_env
source model_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Model export ediliyor..."
yolo export model=yolo11s.pt format=onnx half=True

deactivate

echo "Export tamamlandı: yolo11s.onnx"
