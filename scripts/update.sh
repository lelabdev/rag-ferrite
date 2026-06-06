#!/bin/bash
# update.sh — Mise à jour rag-ferrite
# Usage: ./scripts/update.sh [target]
# target: "aether" (défaut, local) ou "nova"

set -e

TARGET="${1:-aether}"
BINARY="./target/release/rag-ferrite"
SERVICE_NAME="rag-ferrite.service"
SERVICE_DIR="$HOME/services/rag-ferrite"

echo "=== Update rag-ferrite → ${TARGET} ==="

# 1. Build
echo "[1/4] Build..."
cargo build --release 2>&1 | tail -1

# 2. Stop service
echo "[2/4] Stop ${TARGET}..."
if [ "$TARGET" = "aether" ]; then
    systemctl --user stop "$SERVICE_NAME"
else
    ssh "$TARGET" "systemctl --user stop $SERVICE_NAME"
fi
sleep 1

# 3. Copy binary
echo "[3/4] Copy binary..."
if [ "$TARGET" = "aether" ]; then
    cp "$BINARY" "$SERVICE_DIR/rag-ferrite"
else
    scp "$BINARY" "$TARGET:$SERVICE_DIR/rag-ferrite"
fi

# 4. Start service
echo "[4/4] Start ${TARGET}..."
if [ "$TARGET" = "aether" ]; then
    systemctl --user start "$SERVICE_NAME"
else
    ssh "$TARGET" "systemctl --user start $SERVICE_NAME"
fi

sleep 2
echo "✓ Done — ${TARGET} updated"
