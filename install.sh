#!/bin/bash
echo "Installing system packages..."
apt-get install -y aria2 portaudio19-dev -q

echo "Installing Fish Speech..."
cd /workspace/fish-speech
pip install -e . -q

echo "Fixing torchmetrics conflict..."
pip uninstall torchvision torchmetrics -y

echo "Installing requirements..."
pip install -r /workspace/tts-backend/requirements.txt -q

echo "✅ All done!"