#!/bin/bash

echo "🚀 Fish Speech TTS Setup Starting..."

# ─────────────────────────────────────────
# 1. Clone Fish Speech Repo
# ─────────────────────────────────────────
echo "📦 Cloning Fish Speech repo..."
cd /workspace
if [ ! -d "fish-speech" ]; then
    git clone https://github.com/fishaudio/fish-speech.git
fi
cd /workspace/fish-speech
pip install -e . -q
echo "✅ Fish Speech installed!"

# ─────────────────────────────────────────
# 2. Install Dependencies
# ─────────────────────────────────────────
echo "📦 Installing dependencies..."
apt-get install -y aria2 portaudio19-dev -q
pip install fastapi uvicorn python-multipart loguru ormsgpack \
    cachetools einops pyrootutils pydub resampy tiktoken natsort \
    kui hydra-core loralib grpcio "einx[torch]==0.2.2" \
    "datasets==2.18.0" descript-audio-codec lightning tensorboard \
    wandb opencc-python-reimplemented pyaudio torchmetrics==0.11.4 -q
echo "✅ Dependencies installed!"

# ─────────────────────────────────────────
# 3. Download Models
# ─────────────────────────────────────────
echo "📥 Downloading models..."
mkdir -p /workspace/checkpoints/s2-pro
mkdir -p /workspace/checkpoints/fish-speech-1.5
mkdir -p /workspace/storage/voices
mkdir -p /workspace/storage/output

BASE_URL="https://huggingface.co/fishaudio/s2-pro/resolve/main"

# Model files list
declare -A FILES=(
    ["model-00001-of-00002.safetensors"]="$BASE_URL/model-00001-of-00002.safetensors"
    ["model-00002-of-00002.safetensors"]="$BASE_URL/model-00002-of-00002.safetensors"
    ["codec.pth"]="$BASE_URL/codec.pth"
    ["config.json"]="$BASE_URL/config.json"
    ["tokenizer.json"]="$BASE_URL/tokenizer.json"
    ["tokenizer_config.json"]="$BASE_URL/tokenizer_config.json"
    ["special_tokens_map.json"]="$BASE_URL/special_tokens_map.json"
    ["chat_template.jinja"]="$BASE_URL/chat_template.jinja"
    ["model.safetensors.index.json"]="$BASE_URL/model.safetensors.index.json"
)

for FILE in "${!FILES[@]}"; do
    DEST="/workspace/checkpoints/s2-pro/$FILE"
    if [ ! -f "$DEST" ]; then
        echo "⬇️  Downloading $FILE..."
        aria2c --console-log-level=notice --file-allocation=none \
            -x 16 -s 16 -k 1M \
            "${FILES[$FILE]}" \
            -d /workspace/checkpoints/s2-pro \
            -o "$FILE"
    else
        echo "✅ $FILE already exists, skipping..."
    fi
done

echo "✅ All models downloaded!"

# ─────────────────────────────────────────
# 4. Copy API file
# ─────────────────────────────────────────
echo "📋 Copying API backend..."
cp /workspace/tts-backend/api.py /workspace/api.py
echo "✅ API copied!"

echo ""
echo "🎉 Setup Complete! Now run:"
echo "   bash /workspace/tts-backend/start.sh"
