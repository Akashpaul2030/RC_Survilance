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
HTTPS_PORT=8443
PYTHON_VERSION="3.11"
CERT_DIR="/opt/rc-surveillance/certs"

echo "=============================================="
echo " RC Surveillance — Azure VM Deployment"
echo "=============================================="

# ── 1. System packages ────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    git python3 python3-pip python3-venv \
    libsndfile1 ffmpeg curl cmake build-essential \
    python3.11 python3.11-venv python3.11-dev 2>/dev/null || \
apt-get install -y -qq \
    git python3 python3-pip python3-venv \
    libsndfile1 ffmpeg curl cmake build-essential

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
# Prefer python3.11 to avoid onnx build issues on python3.12
PYTHON_BIN=$(command -v python3.11 || command -v python3)
echo "  Using Python: $($PYTHON_BIN --version)"
$PYTHON_BIN -m venv "$APP_DIR/venv"
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

# ── 5a. Generate self-signed SSL certificate ──────
echo "[5/7] Generating self-signed SSL certificate for HTTPS..."
apt-get install -y -qq openssl
mkdir -p "$CERT_DIR"
PUBLIC_IP=$(curl -sf "https://api.ipify.org" 2>/dev/null || echo "localhost")
openssl req -x509 -newkey rsa:2048 -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" -days 3650 -nodes \
    -subj "/CN=${PUBLIC_IP}" \
    -addext "subjectAltName=IP:${PUBLIC_IP}" 2>/dev/null
chmod 600 "$CERT_DIR/key.pem"
echo "  SSL cert generated for IP: ${PUBLIC_IP}"

# ── 5b. systemd service ───────────────────────────
echo "[6/7] Installing systemd service..."
cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=RC Surveillance ONNX Audio Classifier (FastAPI)
After=network.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=${APP_DIR}
Environment="MODEL_DIR=${APP_DIR}/model_onnx_int8"
Environment="ORT_NUM_THREADS=4"
ExecStart=${APP_DIR}/venv/bin/uvicorn app:app --host 0.0.0.0 --port ${HTTPS_PORT} --workers 1 --ssl-keyfile ${CERT_DIR}/key.pem --ssl-certfile ${CERT_DIR}/cert.pem
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

chown -R azureuser:azureuser "$CERT_DIR"
chown -R www-data:www-data "$APP_DIR"
chown -R azureuser:azureuser "$CERT_DIR"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

# ── 7. Health check ───────────────────────────────
echo "[7/7] Waiting for service to start..."
sleep 8
if curl -sfk "https://localhost:${HTTPS_PORT}/health" > /dev/null; then
    echo ""
    echo "=============================================="
    echo " Deployment SUCCESSFUL — HTTPS ENABLED"
    echo "=============================================="
    curl -sk "https://localhost:${HTTPS_PORT}/health" | python3 -m json.tool
    echo ""
    PUBLIC_IP=$(curl -sf "https://api.ipify.org" 2>/dev/null || echo "<vm-public-ip>")
    echo ""
    echo "  Open in phone browser (accept cert warning):"
    echo "  >>> https://${PUBLIC_IP}:${HTTPS_PORT} <<<"
    echo ""
    echo "  NOTE: Azure NSG must allow TCP port ${HTTPS_PORT}"
    echo "  In Azure Portal: VM → Networking → Add inbound rule → Port ${HTTPS_PORT}"
    echo "=============================================="
else
    echo "[ERROR] Service did not start. Check logs:"
    echo "  journalctl -u ${SERVICE_NAME} -n 50 --no-pager"
    systemctl status "$SERVICE_NAME" --no-pager
    exit 1
fi
