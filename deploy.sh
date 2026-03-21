#!/usr/bin/env bash
# deploy.sh — One-shot deployment script for RC Surveillance ONNX classifier
# Run this on the Azure VM via Cloud Shell:
#   bash <(curl -fsSL https://raw.githubusercontent.com/Akashpaul2030/RC_Survilance/claude/onnx-audio-classifier-deploy-b1Mrn/deploy.sh)
# Or after cloning:
#   chmod +x deploy.sh && sudo ./deploy.sh

set -euo pipefail

REPO_URL="https://github.com/Akashpaul2030/RC_Survilance.git"
BRANCH="claude/onnx-audio-classifier-deploy-b1Mrn"
APP_DIR="/opt/rc-surveillance"
SERVICE_NAME="rc-surveillance"
APP_PORT=8000
PYTHON_VERSION="3.11"

echo "=============================================="
echo " RC Surveillance — Azure VM Deployment"
echo "=============================================="

# ── 1. System packages ────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    git python3 python3-pip python3-venv \
    libsndfile1 ffmpeg curl

# ── 2. Clone / update repo ────────────────────────
echo "[2/6] Cloning repository (branch: $BRANCH)..."
if [ -d "$APP_DIR/.git" ]; then
    echo "  Repo already exists — pulling latest..."
    git -C "$APP_DIR" fetch origin "$BRANCH"
    git -C "$APP_DIR" reset --hard "origin/$BRANCH"
else
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$APP_DIR"
fi

# ── 3. Python virtual environment + deps ──────────
echo "[3/6] Setting up Python virtual environment..."
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip -q

echo "  Installing PyTorch (CPU-only, saves ~1 GB vs full torch)..."
"$APP_DIR/venv/bin/pip" install --quiet \
    torch==2.2.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

echo "  Installing remaining dependencies..."
"$APP_DIR/venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

# ── 4. Verify model exists ────────────────────────
echo "[4/6] Checking for ONNX model..."
MODEL_DIR="$APP_DIR/model_onnx_int8"
if [ ! -f "$MODEL_DIR/model_quantized.onnx" ] && [ ! -f "$MODEL_DIR/model.onnx" ]; then
    echo "  [!] Model not found in $MODEL_DIR"
    echo "  Running export_onnx.py to generate it (this takes ~5 minutes)..."
    cd "$APP_DIR"
    "$APP_DIR/venv/bin/python" export_onnx.py
else
    echo "  Model found: $(ls "$MODEL_DIR"/*.onnx)"
fi

# ── 5. systemd service ────────────────────────────
echo "[5/6] Installing systemd service..."
cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=RC Surveillance ONNX Audio Classifier (FastAPI)
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=${APP_DIR}
Environment="MODEL_DIR=${APP_DIR}/model_onnx_int8"
Environment="ORT_NUM_THREADS=4"
ExecStart=${APP_DIR}/venv/bin/uvicorn app:app --host 0.0.0.0 --port ${APP_PORT} --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

chown -R www-data:www-data "$APP_DIR"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

# ── 6. Health check ───────────────────────────────
echo "[6/6] Waiting for service to start..."
sleep 6
if curl -sf "http://localhost:${APP_PORT}/health" > /dev/null; then
    echo ""
    echo "=============================================="
    echo " Deployment SUCCESSFUL"
    echo "=============================================="
    curl -s "http://localhost:${APP_PORT}/health" | python3 -m json.tool
    echo ""
    PUBLIC_IP=$(curl -sf "https://api.ipify.org" 2>/dev/null || echo "<vm-public-ip>")
    echo "  API base URL : http://${PUBLIC_IP}:${APP_PORT}"
    echo "  Health check : http://${PUBLIC_IP}:${APP_PORT}/health"
    echo "  Predict      : POST http://${PUBLIC_IP}:${APP_PORT}/predict  (upload audio file)"
    echo "  Stream (WS)  : ws://${PUBLIC_IP}:${APP_PORT}/stream"
    echo "=============================================="
else
    echo "[ERROR] Service did not start. Check logs:"
    echo "  journalctl -u ${SERVICE_NAME} -n 50 --no-pager"
    systemctl status "$SERVICE_NAME" --no-pager
    exit 1
fi
