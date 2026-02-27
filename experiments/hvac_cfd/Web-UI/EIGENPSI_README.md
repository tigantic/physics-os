# EigenPsi CFD Analysis System

Complete workflow for CFD consulting: Client intake → Simulation → Report

## Quick Start

```bash
# 1. Install dependencies
pip install fastapi uvicorn websockets aiofiles

# 2. Place files in your HyperFOAM project directory
#    your_project/
#    ├── eigenpsi_server.py      ← Server
#    ├── eigenpsi-cockpit.html   ← Operator UI
#    ├── eigenpsi-intake-form.html  ← Client form (optional, served by server)
#    └── hyperfoam/              ← Your solver

# 3. Run
python eigenpsi_server.py
```

Browser opens automatically to `http://localhost:8420`

---

## Workflow

### 1. Client Intake

**Option A: Web Form**
- Send client to `http://localhost:8420/intake` (or host it publicly)
- Client fills form, clicks "Export JSON"
- Client emails you the JSON file

**Option B: Print Form**
- Open `eigenpsi-intake-form.html` in browser
- Print to PDF
- Email PDF to client
- Client fills it out (digital or paper)
- You transcribe to UI (or use OCR → JSON if digital)

### 2. Import to Cockpit

- Open cockpit: `http://localhost:8420`
- Drag & drop the client's JSON file onto the import zone
- All fields auto-populate
- Review, adjust if needed

### 3. Run Simulation

- Click **Run**
- Watch progress in console
- Results appear when complete

### 4. Deliver Report

- Download PDF report from results panel
- Send to client with thermal heatmaps

---

## File Descriptions

| File | Purpose |
|------|---------|
| `eigenpsi_server.py` | FastAPI server - serves UI, runs simulations |
| `eigenpsi-cockpit.html` | Operator dashboard - single page, all fields visible |
| `eigenpsi-intake-form.html` | Client intake form - prints nicely, exports JSON |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Cockpit UI |
| `/intake` | GET | Client intake form |
| `/api/projects` | GET | List all projects |
| `/api/projects` | POST | Create new project |
| `/api/projects/{id}` | GET | Get project details |
| `/api/projects/{id}/run` | POST | Start simulation |
| `/api/projects/{id}/stop` | POST | Stop simulation |
| `/api/projects/{id}/assets/{file}` | GET | Download output file |
| `/ws/logs` | WebSocket | Real-time log streaming |

---

## Data Flow

```
Client fills intake form
         ↓
    Exports JSON
         ↓
You import to cockpit
         ↓
    Validates fields
         ↓
Converts imperial → metric
         ↓
  Saves job_spec.json
         ↓
   Runs hyperfoam.pipeline
         ↓
 Streams logs via WebSocket
         ↓
  Generates report + images
         ↓
    You deliver to client
```

---

## Customization

### Change Port
```bash
python eigenpsi_server.py --port 9000
```

### Change Projects Directory
Edit `PROJECTS_DIR` in `eigenpsi_server.py`

### Modify Intake Form Fields
Edit the HTML in `eigenpsi-intake-form.html` or the embedded version in the server.

### Add Your Branding
Replace "EigenPsi" with your company name in all files.

---

## Troubleshooting

**Server won't start?**
```bash
pip install fastapi uvicorn websockets aiofiles
```

**CUDA not detected?**
- Check `nvidia-smi` works
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Simulation fails?**
- Check `projects/{id}/output/` for error logs
- Verify `hyperfoam` module is importable

**Form fields not importing?**
- Ensure JSON structure matches expected format
- Check browser console for errors
