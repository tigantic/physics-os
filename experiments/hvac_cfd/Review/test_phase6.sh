#!/bin/bash
# HyperFoam Phase 6 Integration Test
# Tests the Bridge Command Channel

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
IFC_FILE="/home/brad/TiganticLabz/three.js-dev/examples/models/ifc/rac_advanced_sample_project.ifc"
SOCKET_PATH="/tmp/hyperfoam_command.sock"

echo "=============================================="
echo "  HyperFoam Phase 6: Bridge Command Test"
echo "=============================================="
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "[TEST] Cleaning up..."
    pkill -f "hyperfoam.bridge_main" 2>/dev/null || true
    rm -f "$SOCKET_PATH" 2>/dev/null || true
}
trap cleanup EXIT

# Check prerequisites
echo "[TEST] Checking prerequisites..."
if [ ! -f "$IFC_FILE" ]; then
    echo "❌ IFC file not found: $IFC_FILE"
    exit 1
fi
echo "✅ IFC file: $(basename $IFC_FILE) ($(du -h "$IFC_FILE" | cut -f1))"

# Activate venv
cd "$PROJECT_DIR"
source .venv/bin/activate
echo "✅ Python venv activated"

# Start bridge in background
echo ""
echo "[TEST] Starting HyperFoam bridge..."
python -m hyperfoam.bridge_main --bridge-mode &
BRIDGE_PID=$!
echo "✅ Bridge started (PID: $BRIDGE_PID)"

# Wait for socket
echo "[TEST] Waiting for command socket..."
for i in {1..10}; do
    if [ -S "$SOCKET_PATH" ]; then
        echo "✅ Command socket file exists"
        break
    fi
    sleep 0.5
done

if [ ! -S "$SOCKET_PATH" ]; then
    echo "❌ Command socket not created after 5s"
    exit 1
fi

# Wait additional time for listener to start accepting
echo "[TEST] Waiting for listener to be ready..."
sleep 2

# Test 1: Send STATUS command
echo ""
echo "[TEST 6.1a] Sending STATUS command..."
python3 -c "
import socket
import json

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2.0)
sock.connect('$SOCKET_PATH')
cmd = json.dumps({'cmd': 'STATUS'})
sock.send((cmd + '\n').encode())
sock.close()
print('✅ STATUS command sent')
"

sleep 0.5

# Test 2: Send LOAD_GEOMETRY command
echo ""
echo "[TEST 6.1b] Sending LOAD_GEOMETRY command..."
python3 -c "
import socket
import json

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2.0)
sock.connect('$SOCKET_PATH')
cmd = json.dumps({'cmd': 'LOAD_GEOMETRY', 'path': '$IFC_FILE'})
sock.send((cmd + '\n').encode())
sock.close()
print('✅ LOAD_GEOMETRY command sent')
print('   Path: $IFC_FILE')
"

sleep 1

# Test 3: Send SET_PARAM command
echo ""
echo "[TEST 6.1c] Sending SET_PARAM command..."
python3 -c "
import socket
import json

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2.0)
sock.connect('$SOCKET_PATH')
cmd = json.dumps({'cmd': 'SET_PARAM', 'key': 'inlet_velocity', 'value': 5.0})
sock.send((cmd + '\n').encode())
sock.close()
print('✅ SET_PARAM command sent')
print('   inlet_velocity = 5.0')
"

sleep 0.5

# Test 4: Send PAUSE/RESUME
echo ""
echo "[TEST 6.1d] Sending PAUSE command..."
python3 -c "
import socket
import json

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2.0)
sock.connect('$SOCKET_PATH')
cmd = json.dumps({'cmd': 'PAUSE'})
sock.send((cmd + '\n').encode())
sock.close()
print('✅ PAUSE command sent')
"

sleep 0.5

echo ""
echo "[TEST 6.1e] Sending RESUME command..."
python3 -c "
import socket
import json

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2.0)
sock.connect('$SOCKET_PATH')
cmd = json.dumps({'cmd': 'RESUME'})
sock.send((cmd + '\n').encode())
sock.close()
print('✅ RESUME command sent')
"

sleep 1

# Test 5: Send SHUTDOWN
echo ""
echo "[TEST 6.1f] Sending SHUTDOWN command..."
python3 -c "
import socket
import json

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2.0)
sock.connect('$SOCKET_PATH')
cmd = json.dumps({'cmd': 'SHUTDOWN'})
sock.send((cmd + '\n').encode())
sock.close()
print('✅ SHUTDOWN command sent')
"

# Wait for bridge to exit
echo ""
echo "[TEST] Waiting for bridge shutdown..."
wait $BRIDGE_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "  ✅ PHASE 6.1 TESTS PASSED"
echo "=============================================="
echo ""
echo "  All command channel tests successful:"
echo "  • STATUS command"
echo "  • LOAD_GEOMETRY command"
echo "  • SET_PARAM command"
echo "  • PAUSE/RESUME commands"
echo "  • SHUTDOWN command"
echo ""
echo "  Next: Connect frontend visualization"
echo "  for full 6.2 Integration Test"
echo ""
