#!/usr/bin/env bash
set -euo pipefail

# rag-ferrite installer — Linux x86_64
# Usage: curl -fsSL https://raw.githubusercontent.com/lelabdev/rag-ferrite/main/install.sh | bash

REPO="lelabdev/rag-ferrite"
BINARY="rag-ferrite"
INSTALL_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/rag-ferrite"
DATA_DIR="$HOME/.local/share/rag-ferrite"
SERVICE_DIR="$HOME/.config/systemd/user"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Checks ---

ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" != "Linux" ]]; then
    error "This installer only supports Linux. You're on $OS."
fi

if [[ "$ARCH" != "x86_64" && "$ARCH" != "aarch64" ]]; then
    error "Unsupported architecture: $ARCH. Only x86_64 and aarch64 are supported."
fi

# --- Get latest release ---

info "Fetching latest release from GitHub..."
LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null | grep '"tag_name"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/')

if [[ -z "$LATEST" ]]; then
    error "Could not fetch latest release. Check your internet connection."
fi

info "Latest version: ${LATEST}"

# --- Download binary ---

mkdir -p "$INSTALL_DIR"

DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${LATEST}/${BINARY}"

info "Downloading ${BINARY} ${LATEST}..."
curl -fsSL "$DOWNLOAD_URL" -o "${INSTALL_DIR}/${BINARY}" || error "Download failed"
chmod +x "${INSTALL_DIR}/${BINARY}"

info "Installed to ${INSTALL_DIR}/${BINARY}"

# --- Config ---

mkdir -p "$CONFIG_DIR"

if [[ -f "${CONFIG_DIR}/config.toml" ]]; then
    warn "Config already exists at ${CONFIG_DIR}/config.toml — skipping"
else
    info "Generating default config..."
    cat > "${CONFIG_DIR}/config.toml" << 'EOF'
[llm]
provider = "ollama"
model = "gemma3:12b"
base_url = "http://localhost:11434"
context_enabled = true
relevance_scoring = true
min_relevance_score = 5.0

[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 1536
# base_url = ""  # defaults to OpenAI
# api_key = ""   # or set OPENAI_API_KEY env var

[reranker]
reranker_type = "disabled"
top_k = 10
preview_chars = 300

[advanced]
chunk_size = 800
quality_threshold = 0.3
query_limit = 10
EOF
    info "Config written to ${CONFIG_DIR}/config.toml"
    warn "Edit config.toml to set your LLM provider and embedding model before running"
fi

# --- Data dir ---

mkdir -p "$DATA_DIR"

# --- Systemd service (optional) ---

setup_service() {
    mkdir -p "$SERVICE_DIR"

    cat > "${SERVICE_DIR}/rag-ferrite.service" << EOF
[Unit]
Description=rag-ferrite RAG engine
After=network.target

[Service]
Type=simple
ExecStart=${INSTALL_DIR}/${BINARY}
WorkingDirectory=${CONFIG_DIR}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload 2>/dev/null || true
    info "Systemd service installed to ${SERVICE_DIR}/rag-ferrite.service"
    info "Enable with:  systemctl --user enable --now rag-ferrite"
}

# --- PATH ---

ensure_path() {
    local shell_rc=""
    if [[ -f "$HOME/.bashrc" ]]; then shell_rc="$HOME/.bashrc"
    elif [[ -f "$HOME/.zshrc" ]]; then shell_rc="$HOME/.zshrc"
    elif [[ -f "$HOME/.profile" ]]; then shell_rc="$HOME/.profile"
    fi

    if [[ -n "$shell_rc" ]] && ! grep -q "$INSTALL_DIR" "$shell_rc" 2>/dev/null; then
        echo "" >> "$shell_rc"
        echo "export PATH=\"\$PATH:${INSTALL_DIR}\"" >> "$shell_rc"
        info "Added ${INSTALL_DIR} to PATH in ${shell_rc}"
        warn "Run 'source ${shell_rc}' or open a new terminal to use rag-ferrite"
    fi
}

# --- Main ---

echo ""
info "=== rag-ferrite ${LATEST} installed ==="
echo ""
echo "  Binary:   ${INSTALL_DIR}/${BINARY}"
echo "  Config:   ${CONFIG_DIR}/config.toml"
echo "  Data:     ${DATA_DIR}/"
echo ""
echo "  Quick start:"
echo "    1. Edit config:   nano ${CONFIG_DIR}/config.toml"
echo "    2. Run:           ${INSTALL_DIR}/${BINARY}"
echo ""

ensure_path

read -p "Install as systemd user service? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    setup_service
fi

echo ""
info "Done."
