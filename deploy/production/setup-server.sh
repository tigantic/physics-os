#!/usr/bin/env bash
# ── Physics OS — Production Server Setup ────────────────────────────
# One-shot provisioning script for a bare Ubuntu 22.04+ server.
#
# Usage:
#   sudo bash deploy/production/setup-server.sh
#
# What it does:
#   1. Creates 'ontic' service user
#   2. Installs system packages (Python 3.12, nginx, certbot)
#   3. Clones repo and creates virtualenv
#   4. Installs Physics OS with [server] extras
#   5. Installs systemd unit + nginx config
#   6. Obtains TLS certificate via Let's Encrypt
#   7. Starts the service
#
# Prerequisites:
#   - Ubuntu 22.04+ or Debian 12+
#   - DNS A record for your domain pointing to this server
#   - Root / sudo access
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────
DOMAIN="${ONTIC_DOMAIN:-api.physics-os.com}"
EMAIL="${ONTIC_CERTBOT_EMAIL:-admin@physics-os.com}"
INSTALL_DIR="/opt/physics-os"
REPO_URL="https://github.com/tigantic/physics-os.git"
BRANCH="main"
PYTHON="python3.12"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Pre-flight ──────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (sudo)."
    exit 1
fi

info "Physics OS — Production Server Setup"
info "Domain:      $DOMAIN"
info "Install dir: $INSTALL_DIR"
echo ""

# ── 1. System packages ─────────────────────────────────────────────
info "Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    software-properties-common \
    curl \
    git \
    build-essential \
    nginx \
    certbot \
    python3-certbot-nginx

# Python 3.12
if ! command -v "$PYTHON" &>/dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
fi

info "System packages installed."

# ── 2. Create service user ─────────────────────────────────────────
if ! id ontic &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir "$INSTALL_DIR" ontic
    info "Created 'ontic' service user."
else
    info "User 'ontic' already exists."
fi

# ── 3. Clone / update repository ───────────────────────────────────
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Updating existing installation..."
    cd "$INSTALL_DIR"
    git fetch origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
else
    info "Cloning repository..."
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# ── 4. Python virtualenv + dependencies ────────────────────────────
info "Setting up Python environment..."
if [[ ! -d "$INSTALL_DIR/venv" ]]; then
    "$PYTHON" -m venv "$INSTALL_DIR/venv"
fi

"$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel setuptools -q
"$INSTALL_DIR/venv/bin/pip" install -e ".[server]" -q

info "Python environment ready."

# ── 5. Directory structure ──────────────────────────────────────────
mkdir -p "$INSTALL_DIR/data" "$INSTALL_DIR/logs"
chown -R ontic:ontic "$INSTALL_DIR"

# ── 6. Environment file ────────────────────────────────────────────
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    API_KEY=$(openssl rand -base64 32 | tr -d '=/+' | head -c 40)
    cat > "$INSTALL_DIR/.env" <<EOF
# Physics OS — Production Configuration
# Generated: $(date -Iseconds)
# See: deploy/production/.env.production.template

# ── Server ──────────────────────────────────────────────────────────
ONTIC_HOST=127.0.0.1
ONTIC_PORT=8000
ONTIC_WORKERS=2
ONTIC_DEBUG=false
ONTIC_LOG_LEVEL=info
ONTIC_CORS_ORIGINS=https://${DOMAIN}

# ── Auth ────────────────────────────────────────────────────────────
ONTIC_REQUIRE_AUTH=true
ONTIC_API_KEYS=${API_KEY}

# ── Compute ─────────────────────────────────────────────────────────
ONTIC_DEVICE=cpu
ONTIC_MAX_N_BITS=14
ONTIC_MAX_N_STEPS=10000
ONTIC_JOB_TIMEOUT_S=300

# ── Billing (shadow mode until Stripe is configured) ────────────────
ONTIC_BILLING_MODE=shadow
ONTIC_STRIPE_SECRET_KEY=
ONTIC_STRIPE_WEBHOOK_SECRET=
ONTIC_STRIPE_PRICE_BUILDER=
ONTIC_STRIPE_PRICE_PRO=
EOF
    chmod 600 "$INSTALL_DIR/.env"
    chown ontic:ontic "$INSTALL_DIR/.env"

    warn "Generated API key: ${API_KEY}"
    warn "Store this key securely — it won't be shown again."
    info "Environment file created at $INSTALL_DIR/.env"
else
    info "Environment file already exists, skipping."
fi

# ── 7. Systemd unit ────────────────────────────────────────────────
info "Installing systemd service..."
cp "$INSTALL_DIR/deploy/production/physics-os.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable physics-os

# ── 8. Nginx config ────────────────────────────────────────────────
info "Configuring nginx..."
# Update domain in nginx config
sed "s/api\.physics-os\.com/${DOMAIN}/g" \
    "$INSTALL_DIR/deploy/production/nginx.conf" \
    > /etc/nginx/sites-available/physics-os

ln -sf /etc/nginx/sites-available/physics-os /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test config before reload
nginx -t
systemctl reload nginx

# ── 9. TLS certificate ─────────────────────────────────────────────
info "Obtaining TLS certificate..."
certbot --nginx \
    -d "$DOMAIN" \
    --email "$EMAIL" \
    --agree-tos \
    --non-interactive \
    --redirect \
    || warn "Certbot failed — TLS will need manual setup."

# ── 10. Start service ──────────────────────────────────────────────
info "Starting Physics OS..."
systemctl start physics-os

# Wait for startup
sleep 3

if systemctl is-active --quiet physics-os; then
    info "Physics OS is running!"
else
    error "Service failed to start. Check: journalctl -u physics-os -n 50"
    exit 1
fi

# ── Health check ────────────────────────────────────────────────────
sleep 2
if curl -sf http://127.0.0.1:8000/v1/health > /dev/null 2>&1; then
    info "Health check passed ✓"
else
    warn "Health check failed — service may still be starting."
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
info "Physics OS deployed successfully!"
echo ""
echo "  API endpoint:  https://${DOMAIN}/v1/"
echo "  Health check:  https://${DOMAIN}/v1/health"
echo "  Service logs:  journalctl -u physics-os -f"
echo "  Config file:   ${INSTALL_DIR}/.env"
echo ""
echo "  Next steps:"
echo "    1. Configure Stripe keys in ${INSTALL_DIR}/.env"
echo "    2. Set ONTIC_BILLING_MODE=live"
echo "    3. sudo systemctl restart physics-os"
echo "═══════════════════════════════════════════════════════════════"
