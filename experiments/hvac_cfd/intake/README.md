# HyperFOAM Universal Intake System

Enterprise-grade document ingestion for CFD simulation configuration.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r intake/requirements.txt

# Optional: Install Tesseract OCR for image text extraction
sudo apt install tesseract-ocr

# Launch the intake system
./run_intake.sh
```

Then open http://localhost:8501 in your browser.

## 📁 Supported File Types

| Format | Description | Extraction Capabilities |
|--------|-------------|-------------------------|
| **PDF** | Blueprints, specifications | Text, dimensions, equipment schedules |
| **PNG/JPG** | Blueprint images | OCR text, scale detection |
| **IFC** | BIM models | Full geometry, spaces, HVAC equipment |
| **DOCX** | Specification documents | Design conditions, equipment data |
| **JSON** | Direct job_spec | Full field mapping |

## 📏 Measurement Units

The system supports automatic unit detection and conversion:

### Imperial (US Customary)
- Length: feet (ft), inches (in), feet-inches (10'-6")
- Temperature: Fahrenheit (°F)
- Airflow: CFM (cubic feet per minute)
- Velocity: ft/min (FPM)
- Pressure: inWG (inches water gauge)

### Metric (SI)
- Length: meters (m), centimeters (cm), millimeters (mm)
- Temperature: Celsius (°C)
- Airflow: m³/s, m³/h, L/s
- Velocity: m/s
- Pressure: Pascal (Pa)

All values are converted to SI units internally for the CFD solver.

## 🔄 Workflow

```
┌─────────────────┐
│  Upload Files   │  Drop PDF, PNG, IFC, DOCX, JSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Auto-Extraction │  OCR, IfcOpenShell, regex parsing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Review & Edit   │  Confirm extracted values
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Configure      │  Fill mandatory/recommended fields
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │  Create job_spec.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Submit         │  Run CFD simulation
└─────────────────┘
```

## 📋 Field Categories

### 🔴 Mandatory (Required for simulation)
- Project name
- Room dimensions (length, width, height)
- Number of supply diffusers
- Total supply airflow
- Supply air temperature

### 🟡 Recommended (Improves accuracy)
- HVAC system type
- Return grille count
- Diffuser type
- Design occupancy
- Lighting/equipment loads
- Temperature setpoints

### ⚪ Optional (Advanced/fine-tuning)
- Turbulence model
- Grid resolution
- Simulation duration
- Custom geometry file

### ✅ Compliance (Standards & regulations)
- ADPI target (ASHRAE 55)
- PPD limit (ISO 7730)
- Max air velocity
- CO₂ limits (ASHRAE 62.1)
- Ventilation standard

## 🏗️ Architecture

```
intake/
├── __init__.py          # Package exports
├── app.py               # Streamlit application
├── schema.py            # Field definitions
├── units.py             # Unit conversion
├── job_spec.py          # Job spec generator
├── requirements.txt     # Dependencies
└── extractors/
    ├── __init__.py      # Base extractor class
    ├── pdf_extractor.py
    ├── image_extractor.py
    ├── ifc_extractor.py
    └── document_extractor.py
```

## 🔧 Configuration

### Environment Variables

```bash
# Custom port
export STREAMLIT_PORT=8502

# Disable telemetry
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Tesseract OCR Configuration

For best OCR results on blueprints:

```bash
# Install additional language packs if needed
sudo apt install tesseract-ocr-eng

# Set custom config
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

## 📊 Output Format

The generated `job_spec.json` follows this structure:

```json
{
  "version": "2.0",
  "project": { ... },
  "geometry": {
    "dimensions": { "length": 9.144, "width": 6.096, "height": 3.048 },
    "units": "meters",
    "original_units": "ft"
  },
  "hvac": { ... },
  "sources": { ... },
  "loads": { ... },
  "solver": { ... },
  "targets": { ... },
  "compliance": { ... }
}
```

## 🔗 Integration with HyperFOAM

After generating the job spec:

```bash
# Run simulation
python -m hyperfoam run /path/to/job_spec.json

# Or use the Qt GUI
./build/HyperFOAM --load /path/to/job_spec.json
```

## 🐛 Troubleshooting

### PDF Extraction Issues
```bash
# Install PyMuPDF
pip install pymupdf

# For scanned PDFs, ensure Tesseract is installed
sudo apt install tesseract-ocr
```

### IFC Import Errors
```bash
# IfcOpenShell requires specific Python version
pip install ifcopenshell

# Or build from source for latest features
```

### OCR Quality Issues
- Ensure blueprint is high resolution (300 DPI minimum)
- Use clean, high-contrast images
- Remove annotations/markups before scanning

## 📄 License

Proprietary - HyperFOAM Team

## 🤝 Contributing

See CONTRIBUTING.md in the root directory.
