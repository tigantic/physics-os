# HyperFOAM Backend

FastAPI server providing REST endpoints for the Physics OS UI.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn

# Run the server
uvicorn backend.main:app --reload --port 8000

# Or run directly
python -m backend.main
```

## API Endpoints

### Simulations
- `GET /api/v1/simulations` - List simulations
- `POST /api/v1/simulations` - Create simulation
- `GET /api/v1/simulations/{id}` - Get simulation
- `POST /api/v1/simulations/{id}/start` - Start simulation
- `POST /api/v1/simulations/{id}/pause` - Pause simulation
- `POST /api/v1/simulations/{id}/stop` - Stop simulation
- `DELETE /api/v1/simulations/{id}` - Delete simulation
- `GET /api/v1/simulations/{id}/residuals` - Get residuals

### Meshes
- `GET /api/v1/meshes` - List meshes
- `POST /api/v1/meshes` - Create mesh
- `GET /api/v1/meshes/{id}` - Get mesh details
- `POST /api/v1/meshes/{id}/patches` - Add boundary patch

### System
- `GET /api/v1/system/status` - System health/status
- `GET /api/v1/system/gpus` - GPU information

### WebSocket
- `ws://localhost:8000/ws` - Real-time simulation updates

## Development

The server includes demo data for development:
- 3 sample meshes (Office, Data Center, Conference Room)
- 3 sample simulations (Completed, Running, Pending)
