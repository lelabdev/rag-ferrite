#!/bin/bash
# update.sh — Déploiement rag-ferrite (téléchargement depuis GitHub Releases)
# Appelé par: rag-ferrite update
# Le script vit à côté du binaire dans ~/services/rag-ferrite/

set -e

SERVICE_NAME="rag-ferrite.service"
BINARY_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY_PATH="${BINARY_DIR}/rag-ferrite"
REPO="lelabdev/rag-ferrite"
TMP_PATH="${BINARY_DIR}/rag-ferrite.new"

echo "=== rag-ferrite update ==="

# 1. Stop service
echo "[1/4] Stopping service..."
systemctl --user stop "${SERVICE_NAME}" 2>/dev/null || true

# Wait until stopped
for i in $(seq 1 10); do
    if ! systemctl --user is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

if systemctl --user is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
    echo "✗ Service still running after 5s, aborting"
    exit 1
fi
echo "✓ Stopped"

# 2. Download from GitHub Releases
echo "[2/4] Downloading latest release..."
if ! curl -sL "https://github.com/${REPO}/releases/latest/download/rag-ferrite" -o "${TMP_PATH}"; then
    echo "✗ Download failed"
    exit 1
fi

if [ ! -s "${TMP_PATH}" ]; then
    echo "✗ Downloaded file is empty"
    rm -f "${TMP_PATH}"
    exit 1
fi
echo "✓ Downloaded"

# 3. Replace binary
echo "[3/4] Replacing binary..."
chmod +x "${TMP_PATH}"
mv "${TMP_PATH}" "${BINARY_PATH}"
echo "✓ Replaced"

# 4. Restart service
echo "[4/4] Starting service..."
systemctl --user start "${SERVICE_NAME}"
sleep 2

if systemctl --user is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
    echo "✓ Service running"
else
    echo "✗ Service failed to start"
    exit 1
fi

echo "=== Done ==="
