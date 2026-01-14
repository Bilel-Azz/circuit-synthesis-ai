#!/bin/bash
# Script de fix pour installer unzip et continuer

ssh ubuntu@57.128.57.31 << 'ENDSSH'
set -e

echo "Installation unzip..."
sudo apt update -qq
sudo apt install -y unzip

echo "Décompression du code..."
cd ~
unzip -o circuit_gnn_colab.zip

echo "Création des dossiers..."
mkdir -p circuit_synthesis_gnn/outputs/data
mv gnn_750k.pt circuit_synthesis_gnn/outputs/data/

echo "Installation Python..."
sudo apt install -y python3.10 python3.10-venv python3-pip

echo "Création environnement virtuel..."
python3.10 -m venv venv
source venv/bin/activate

echo "Installation PyTorch + CUDA..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installation dépendances..."
pip install --quiet numpy matplotlib tqdm

echo "Test GPU..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=== Installation terminée ==="

ENDSSH
