"""
HVAC Document Ingestor
======================

Extracts data from PDFs, Excel files, and text documents.
Returns UI State objects with {value, source, status} for frontend validation.

PUBLIC API (Article V, Section 5.1):
------------------------------------
    HVACDocumentParser.parse(file_path: str) -> Dict[str, Any]
        Main entry point for file-based ingestion.
        
    HVACDocumentParser.parse_bytes(data: bytes, filename: str) -> Dict[str, Any]
        Parse from uploaded file bytes (for web uploads).
        
    HVACDocumentParser.parse_text_content(text: str) -> Dict[str, Any]
        Parse from raw text content (for testing or direct text input).

RETURN STRUCTURE:
-----------------
    {
        "success": bool,
        "raw_text": str,  # Original extracted text
        "fields": {
            "field_name": {
                "value": Any,
                "source": str,  # See FieldSource enum
                "status": str,  # See FieldStatus enum
                "confidence": float,  # 0.0-1.0
                "original_text": str  # Matched text snippet
            }
        },
        "summary": {
            "total_fields": int,
            "extracted": int,
            "required": int,
            "review": int,
            "ready_to_submit": bool
        }
    }

STATUS TYPES (UI Color Coding):
-------------------------------
    - "required" (🔴 Red): Field is missing, user MUST fill
    - "review" (🟡 Yellow): Field was extracted, user should verify
    - "confirmed" (🟢 Green): User has validated the value
    - "auto_filled" (⚪ Grey): Default or industry standard value

ERROR RETURNS:
--------------
    On failure, returns: {"error": "<actionable error message>"}
    Error messages include guidance per Article V, Section 5.4.

CONSTITUTION COMPLIANCE:
------------------------
    - Article III, Section 3.4: All data validated at boundary
    - Article V, Section 5.1: All public APIs documented
    - Article V, Section 5.4: Actionable error messages
    - Article VII: No stubs, no placeholders, working code only
"""

import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .logger import get_logger

# Module logger
logger = get_logger(__name__)


class FieldStatus(Enum):
    """Field validation status for UI rendering."""
    REQUIRED = "required"      # Red - Must fill
    REVIEW = "review"          # Yellow - Extracted, needs verification
    AUTO_FILLED = "auto_filled"  # Grey - Default value
    CONFIRMED = "confirmed"    # Green - User verified


class FieldSource(Enum):
    """Where the field value came from."""
    MISSING = "missing"
    EXTRACTED = "extracted"
    DEFAULT = "default"
    INDUSTRY_STANDARD = "industry_standard"
    USER_INPUT = "user_input"


@dataclass
class UIField:
    """A single field in the UI State."""
    value: Any
    source: str
    status: str
    confidence: float = 1.0
    original_text: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


class HVACDocumentParser:
    """
    The Ingestor.
    
    Parses PDF/Excel/Text documents and returns UI State for the Staging Area.
    """
    
    def __init__(self):
        # Default values (The "Safety Net")
        self.defaults = {
            "ceiling_height_ft": 9.0,
            "supply_temp_f": 55.0,
            "diffuser_size_in": [24, 24],
            "return_pressure_pa": 0.0,
        }
        
        # Industry standards for auto-fill
        self.industry_standards = {
            "supply_temp_f": 55.0,  # Standard HVAC supply temp
            "return_temp_f": 75.0,  # Standard return temp
            "air_changes_per_hour": 6,  # Office standard
        }
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point for file-based document ingestion.
        
        Args:
            file_path: Absolute or relative path to the document file.
                       Supported formats: PDF, XLSX, XLS, CSV, TXT, MD
        
        Returns:
            UI State dictionary with extracted fields, or error dict.
            See module docstring for complete return structure.
        
        Raises:
            No exceptions raised - all errors returned in {"error": ...} format
            per Article III, Section 3.2 (graceful failure).
        """
        start_time = time.time()
        path = Path(file_path)
        
        logger.info("Starting document parse", extra={
            "file_name": path.name,
            "file_size_kb": path.stat().st_size // 1024 if path.exists() else 0,
        })
        
        if not path.exists():
            logger.warning("File not found", extra={"path": str(file_path)})
            return {
                "error": f"File not found: {file_path}. "
                         f"Check that the file path is correct and the file exists."
            }
        
        if not path.is_file():
            logger.warning("Path is not a file", extra={"path": str(file_path)})
            return {
                "error": f"Path is not a file: {file_path}. "
                         f"Please provide a path to a document file, not a directory."
            }
        
        ext = path.suffix.lower()
        supported_formats = {
            ".pdf": self._parse_pdf,
            ".xlsx": self._parse_excel,
            ".xls": self._parse_excel,
            ".csv": self._parse_csv,
            ".txt": self._parse_text,
            ".md": self._parse_text,
            ".png": self._parse_blueprint_image,
            ".jpg": self._parse_blueprint_image,
            ".jpeg": self._parse_blueprint_image,
            ".tiff": self._parse_blueprint_image,
            ".bmp": self._parse_blueprint_image,
        }
        
        try:
            parser_func = supported_formats.get(ext, self._parse_text)
            result = parser_func(path)
            
            duration_ms = (time.time() - start_time) * 1000
            fields_extracted = result.get("summary", {}).get("extracted", 0)
            
            logger.info("Document parse complete", extra={
                "file_name": path.name,
                "format": ext,
                "fields_extracted": fields_extracted,
                "duration_ms": round(duration_ms, 2),
                "success": "error" not in result,
            })
            
            return result
            
        except MemoryError:
            logger.error("Memory error parsing file", extra={"file_name": path.name})
            return {
                "error": f"File too large to process: {path.name}. "
                         f"Maximum supported file size is ~50MB. "
                         f"Try splitting the document or extracting relevant pages."
            }
        except PermissionError:
            return {
                "error": f"Permission denied reading: {path.name}. "
                         f"Check file permissions or try copying the file to a different location."
            }
        except Exception as e:
            return {
                "error": f"Parse failed for {path.name}: {str(e)}. "
                         f"If this is a scanned PDF, try converting to searchable PDF first. "
                         f"For Excel files, ensure the file is not password-protected."
            }
    
    def parse_bytes(self, data: bytes, filename: str) -> Dict[str, Any]:
        """Parse from uploaded file bytes."""
        import tempfile
        import os
        
        ext = Path(filename).suffix.lower()
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            temp_path = f.name
        
        try:
            result = self.parse(temp_path)
        finally:
            os.unlink(temp_path)
        
        return result
    
    def parse_text_content(self, text: str) -> Dict[str, Any]:
        """Parse from raw text content (for testing or direct text input)."""
        extracted = self._scan_text_for_patterns(text)
        return self._build_ui_payload(extracted)
    
    # =========================================================================
    # File Type Handlers
    # =========================================================================
    
    def _parse_pdf(self, path: Path) -> Dict[str, Any]:
        """Extract text from PDF using pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            return {"error": "pdfplumber not installed. Run: pip install pdfplumber"}
        
        full_text = ""
        
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                full_text += " | ".join(str(c) for c in row if c) + "\n"
        except Exception as e:
            return {"error": f"Could not read PDF: {str(e)}"}
        
        # If no text extracted, try OCR
        if not full_text.strip():
            ocr_result = self._try_ocr_pdf(path)
            if ocr_result.get("success"):
                full_text = ocr_result["text"]
                logger.info("PDF text extracted via OCR", extra={
                    "file": path.name,
                    "text_length": len(full_text),
                })
            elif "error" in ocr_result:
                # OCR failed or not available - return helpful message
                return {
                    "error": f"No text found in PDF (appears to be scanned). {ocr_result['error']}"
                }
            else:
                return {"error": "No text found in PDF. May be a scanned image. Install pytesseract for OCR."}
        
        extracted = self._scan_text_for_patterns(full_text)
        result = self._build_ui_payload(extracted, raw_text=full_text)
        return result
    
    def _try_ocr_pdf(self, path: Path) -> Dict[str, Any]:
        """
        Attempt OCR extraction from PDF using pytesseract.
        
        Returns:
            {"success": True, "text": "..."} on success
            {"success": False, "error": "..."} on failure
        """
        # Check for pytesseract
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return {
                "success": False,
                "error": "OCR not available. Install: pip install pytesseract pillow"
            }
        
        # Check for pdf2image (needed to convert PDF pages to images)
        try:
            from pdf2image import convert_from_path
        except ImportError:
            return {
                "success": False,
                "error": "PDF-to-image conversion not available. Install: pip install pdf2image"
            }
        
        try:
            # Convert PDF to images
            images = convert_from_path(path, dpi=200)
            full_text = ""
            
            for i, image in enumerate(images):
                # Run OCR on each page
                page_text = pytesseract.image_to_string(image)
                if page_text.strip():
                    full_text += f"\n--- Page {i + 1} ---\n{page_text}\n"
            
            if full_text.strip():
                return {"success": True, "text": full_text}
            else:
                return {"success": False, "error": "OCR found no readable text"}
        
        except pytesseract.TesseractNotFoundError:
            return {
                "success": False,
                "error": "Tesseract not installed. Install tesseract-ocr system package."
            }
        except Exception as e:
            return {"success": False, "error": f"OCR failed: {str(e)}"}
    
    def _parse_excel(self, path: Path) -> Dict[str, Any]:
        """Extract data from Excel using pandas."""
        try:
            import pandas as pd
        except ImportError:
            return {"error": "pandas not installed. Run: pip install pandas openpyxl"}
        
        try:
            # Read all sheets
            xlsx = pd.ExcelFile(path)
            all_data = {}
            full_text = ""
            all_rooms = []  # Multi-room support
            
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                
                # Convert to text for pattern matching
                full_text += f"\n=== {sheet_name} ===\n"
                full_text += df.to_string() + "\n"
                
                # Extract ALL rooms from this sheet
                rooms_from_sheet = self._extract_all_rooms_from_df(df, sheet_name)
                all_rooms.extend(rooms_from_sheet)
                
                # Legacy: also do single-row mapping for backwards compatibility
                self._map_excel_columns(df, all_data)
            
            # Merge extracted data with pattern-matched data
            extracted = self._scan_text_for_patterns(full_text)
            extracted.update(all_data)
            
            result = self._build_ui_payload(extracted, raw_text=full_text)
            
            # Add multi-room data if we found multiple rooms
            if len(all_rooms) > 1:
                result["multi_room"] = {
                    "enabled": True,
                    "room_count": len(all_rooms),
                    "rooms": all_rooms,
                    "selected_index": 0,  # Default to first room
                }
                logger.info("Multi-room extraction", extra={
                    "room_count": len(all_rooms),
                    "room_names": [r.get("room_name", f"Room {i+1}") for i, r in enumerate(all_rooms)]
                })
            elif len(all_rooms) == 1:
                result["multi_room"] = {
                    "enabled": False,
                    "room_count": 1,
                    "rooms": all_rooms,
                    "selected_index": 0,
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Could not read Excel: {str(e)}"}
    
    def _extract_all_rooms_from_df(self, df, sheet_name: str) -> List[Dict[str, Any]]:
        """
        Extract ALL rooms from a DataFrame (Excel schedule).
        
        Returns list of room dicts, each with fields like:
        {
            "room_name": "Conference A",
            "width_ft": 30,
            "length_ft": 25,
            "height_ft": 10,
            "airflow_cfm": 1500,
            "supply_temp_f": 55,
            ...
        }
        """
        import pandas as pd  # Must import here - called from _parse_excel/_parse_csv
        
        rooms = []
        
        # Column name patterns -> our field names
        column_mappings = {
            r'room|space|zone|name': 'room_name',
            r'width|room.*w': 'width_ft',
            r'length|room.*l|depth': 'length_ft',
            r'height|ceiling|h\b': 'height_ft',
            r'cfm|airflow|flow': 'airflow_cfm',
            r'supply.*temp|sat|temp.*f': 'supply_temp_f',
            r'velocity|vel|fpm': 'velocity_fpm',
            r'diffuser|vent|qty|quantity': 'vent_count',
            r'heat|load|watts|btu': 'heat_load',
        }
        
        # Map column names to our field names
        col_to_field = {}
        for col in df.columns:
            col_lower = str(col).lower()
            for pattern, field in column_mappings.items():
                if re.search(pattern, col_lower):
                    col_to_field[col] = field
                    break
        
        # If we don't have at least width or CFM column, skip
        has_dimension = any(f in col_to_field.values() for f in ['width_ft', 'length_ft'])
        has_cfm = 'airflow_cfm' in col_to_field.values()
        
        if not (has_dimension or has_cfm):
            return rooms
        
        # Extract each row as a room
        for idx, row in df.iterrows():
            room_data = {
                "source_sheet": sheet_name,
                "source_row": idx,
            }
            
            has_data = False
            for col, field in col_to_field.items():
                value = row[col]
                if pd.notna(value):
                    # Handle room_name as string
                    if field == 'room_name':
                        room_data[field] = str(value).strip()
                        has_data = True
                    else:
                        # Try to convert to float
                        try:
                            room_data[field] = float(value)
                            has_data = True
                        except (ValueError, TypeError):
                            pass
            
            # Only add if we got some actual data
            if has_data and (room_data.get('width_ft') or room_data.get('airflow_cfm')):
                # Generate room name if not provided
                if 'room_name' not in room_data:
                    room_data['room_name'] = f"Room {idx + 1}"
                rooms.append(room_data)
        
        return rooms
    
    def _parse_csv(self, path: Path) -> Dict[str, Any]:
        """Extract data from CSV using pandas with multi-room support."""
        try:
            import pandas as pd
        except ImportError:
            return {"error": "pandas not installed"}
        
        try:
            df = pd.read_csv(path)
            full_text = df.to_string()
            
            # Extract all rooms from CSV
            all_rooms = self._extract_all_rooms_from_df(df, "CSV")
            
            all_data = {}
            self._map_excel_columns(df, all_data)
            
            extracted = self._scan_text_for_patterns(full_text)
            extracted.update(all_data)
            
            result = self._build_ui_payload(extracted, raw_text=full_text)
            
            # Add multi-room data if found
            if len(all_rooms) > 1:
                result["multi_room"] = {
                    "enabled": True,
                    "room_count": len(all_rooms),
                    "rooms": all_rooms,
                    "selected_index": 0,
                }
                logger.info("CSV multi-room extraction", extra={
                    "room_count": len(all_rooms),
                    "file": path.name,
                })
            elif len(all_rooms) == 1:
                result["multi_room"] = {
                    "enabled": False,
                    "room_count": 1,
                    "rooms": all_rooms,
                    "selected_index": 0,
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Could not read CSV: {str(e)}"}
    
    def _parse_text(self, path: Path) -> Dict[str, Any]:
        """Extract data from plain text/markdown."""
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return {"error": f"Could not read file: {str(e)}"}
        
        extracted = self._scan_text_for_patterns(text)
        return self._build_ui_payload(extracted, raw_text=text)
    
    def _parse_blueprint_image(self, path: Path) -> Dict[str, Any]:
        """
        Extract room dimensions from architectural blueprint images.
        
        Uses OCR to extract text annotations from blueprint images,
        then applies pattern matching to find dimensions.
        
        Supports: PNG, JPG, JPEG, TIFF, BMP
        
        Article III §3.2: Graceful fallback when dependencies unavailable.
        """
        # Check for PIL
        try:
            from PIL import Image
        except ImportError:
            return {
                "error": "Image processing not available. Install: pip install pillow"
            }
        
        # Check for pytesseract (OCR)
        try:
            import pytesseract
        except ImportError:
            return {
                "error": "OCR not available for blueprint analysis. "
                         "Install: pip install pytesseract (and tesseract-ocr system package)"
            }
        
        try:
            # Open and preprocess image
            img = Image.open(path)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get image dimensions for context
            img_width, img_height = img.size
            
            # Run OCR to extract text from blueprint
            ocr_text = pytesseract.image_to_string(img)
            
            if not ocr_text.strip():
                return {
                    "error": "No text found in blueprint image. "
                             "Ensure the image contains readable dimension annotations. "
                             "For best results, use high-resolution scans (300+ DPI)."
                }
            
            logger.info("Blueprint OCR extraction", extra={
                "file": path.name,
                "image_size": f"{img_width}x{img_height}",
                "text_length": len(ocr_text),
            })
            
            # Extract dimensions using our pattern matching
            extracted = self._scan_text_for_patterns(ocr_text)
            
            # Also try to extract scale if present (e.g., "1/4" = 1'-0"")
            scale_match = re.search(
                r'scale[:\s]*(\d+)[/\"]?\s*=\s*(\d+)[\'′\-]',
                ocr_text, re.IGNORECASE
            )
            if scale_match:
                extracted['blueprint_scale'] = f"{scale_match.group(1)}\" = {scale_match.group(2)}'"
            
            result = self._build_ui_payload(extracted, raw_text=ocr_text)
            result["source_type"] = "blueprint_image"
            result["image_dimensions"] = f"{img_width}x{img_height}"
            
            return result
            
        except pytesseract.TesseractNotFoundError:
            return {
                "error": "Tesseract OCR engine not installed. "
                         "Install the tesseract-ocr system package: "
                         "Ubuntu/Debian: sudo apt install tesseract-ocr | "
                         "macOS: brew install tesseract"
            }
        except Exception as e:
            return {
                "error": f"Blueprint analysis failed: {str(e)}. "
                         f"Ensure the image is a valid format and not corrupted."
            }

    def select_room(self, parse_result: Dict[str, Any], room_index: int) -> Dict[str, Any]:
        """
        Select a specific room from multi-room parse results.
        
        Updates the parse result to populate main fields with the selected room's data.
        This enables UI room selection from Excel/CSV schedules with multiple rooms.
        
        Args:
            parse_result: Result from parse() or parse_bytes() with multi_room data
            room_index: Index of room to select (0-based)
        
        Returns:
            Updated parse result with selected room's data in main fields
        """
        if "multi_room" not in parse_result:
            return parse_result
        
        rooms = parse_result["multi_room"].get("rooms", [])
        if room_index < 0 or room_index >= len(rooms):
            logger.warning("Invalid room index for select_room", extra={
                "index": room_index,
                "max": len(rooms) - 1
            })
            return parse_result
        
        selected = rooms[room_index]
        parse_result["multi_room"]["selected_index"] = room_index
        
        # Map room data fields to UI payload fields
        field_mapping = {
            "room_name": "room_name",
            "width_ft": "room_width",
            "length_ft": "room_length",
            "height_ft": "room_height",
            "airflow_cfm": "inlet_cfm",
            "supply_temp_f": "supply_temp",
            "velocity_fpm": "inlet_velocity",
            "vent_count": "vent_count",
            "heat_load": "heat_load",
        }
        
        for room_field, ui_field in field_mapping.items():
            if room_field in selected and ui_field in parse_result.get("fields", {}):
                parse_result["fields"][ui_field]["value"] = selected[room_field]
                parse_result["fields"][ui_field]["status"] = "review"
                parse_result["fields"][ui_field]["source"] = "extracted"
        
        logger.info("Room selected from multi-room result", extra={
            "room_index": room_index,
            "room_name": selected.get("room_name", f"Room {room_index + 1}"),
        })
        
        return parse_result
    
    # =========================================================================
    # Pattern Extraction
    # =========================================================================
    
    def _map_excel_columns(self, df, data: dict):
        """Map Excel column names to our field names."""
        column_mappings = {
            # Column name patterns -> our field name
            r'width|room.*w': 'width_ft',
            r'length|room.*l': 'length_ft',
            r'height|ceiling|h\b': 'height_ft',
            r'cfm|airflow|flow': 'airflow_cfm',
            r'supply.*temp|sat': 'supply_temp_f',
            r'velocity|vel': 'velocity_fpm',
        }
        
        for col in df.columns:
            col_lower = str(col).lower()
            for pattern, field in column_mappings.items():
                if re.search(pattern, col_lower):
                    # Get first non-null value
                    values = df[col].dropna()
                    if len(values) > 0:
                        try:
                            data[field] = float(values.iloc[0])
                        except (ValueError, TypeError):
                            pass
    
    def _scan_text_for_patterns(self, text: str) -> Dict[str, Any]:
        """
        The "Guess" - Extract values using regex patterns.
        """
        data = {}
        
        # Normalize text
        text_lower = text.lower()
        text_norm = re.sub(r'\s+', ' ', text)
        
        # =====================================================================
        # PROJECT INFO
        # =====================================================================
        
        # Project Name
        project_patterns = [
            r'project[:\s]+([A-Za-z][A-Za-z0-9\s\-_]+?)(?:\s*Client|\s*Date|\n|$)',
            r'project\s*name[:\s]+([^\n]+)',
            r'job[:\s]+([^\n]+)',
        ]
        for pattern in project_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['project_name'] = match.group(1).strip()[:100]
                break
        
        # Room Name
        room_patterns = [
            r'room[:\s]+([^\n,]+)',
            r'space[:\s]+([^\n,]+)',
            r'zone[:\s]+([^\n,]+)',
        ]
        for pattern in room_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) < 50 and not re.match(r'^\d+$', name):
                    data['room_name'] = name
                    break
        
        # =====================================================================
        # GEOMETRY - Room Dimensions
        # =====================================================================
        
        # LaTeX format: $3.66\text{m} \times 2.74\text{m} \times 4.57\text{m}$
        latex_dim_3d = re.search(
            r'\$(\d+\.?\d*)\s*\\text\{m\}\s*\\times\s*(\d+\.?\d*)\s*\\text\{m\}\s*\\times\s*(\d+\.?\d*)\s*\\text\{m\}\$',
            text
        )
        if latex_dim_3d:
            # Convert meters to feet
            data['width_ft'] = float(latex_dim_3d.group(1)) / 0.3048
            data['length_ft'] = float(latex_dim_3d.group(2)) / 0.3048
            data['height_ft'] = float(latex_dim_3d.group(3)) / 0.3048
        
        # LaTeX format (2D only): $3.66\text{m} \times 2.74\text{m}$ 
        if 'width_ft' not in data:
            latex_dim_2d = re.search(
                r'\$(\d+\.?\d*)\s*\\text\{m\}\s*\\times\s*(\d+\.?\d*)\s*\\text\{m\}\$',
                text
            )
            if latex_dim_2d:
                data['width_ft'] = float(latex_dim_2d.group(1)) / 0.3048
                data['length_ft'] = float(latex_dim_2d.group(2)) / 0.3048
                # Check for separate ceiling height in LaTeX
                latex_height = re.search(r'\$(\d+\.?\d*)\s*\\text\{m\}\$\s*ceiling', text)
                if latex_height:
                    data['height_ft'] = float(latex_height.group(1)) / 0.3048
        
        # Look for "X ft × Y ft × Z ft" format with explicit units
        if 'width_ft' not in data:
            dims_ft_explicit = re.search(
                r'(\d{1,3}(?:\.\d+)?)\s*ft\s*[×xX]\s*(\d{1,3}(?:\.\d+)?)\s*ft\s*[×xX]\s*(\d{1,3}(?:\.\d+)?)\s*ft',
                text, re.IGNORECASE
            )
            if dims_ft_explicit:
                data['width_ft'] = float(dims_ft_explicit.group(1))
                data['length_ft'] = float(dims_ft_explicit.group(2))
                data['height_ft'] = float(dims_ft_explicit.group(3))
        
        # Standard dimension format: 12x15x10 (only use if values are reasonable room sizes > 5)
        if 'width_ft' not in data:
            dims_match = re.search(
                r'\b(\d{1,3}(?:\.\d+)?)\s*[\'"]?\s*[xX×]\s*(\d{1,3}(?:\.\d+)?)\s*[\'"]?\s*[xX×]\s*(\d{1,3}(?:\.\d+)?)\s*[\'"]?\b',
                text
            )
            if dims_match:
                w, l, h = float(dims_match.group(1)), float(dims_match.group(2)), float(dims_match.group(3))
                # Only accept if these look like room dimensions (> 5 ft, < 200 ft)
                if all(5 <= x <= 200 for x in (w, l, h)):
                    data['width_ft'] = w
                    data['length_ft'] = l
                    data['height_ft'] = h
        
        # 2D dimension format: 20x30 (width x length, uses default height)
        # Common in quick specs and informal documents
        if 'width_ft' not in data:
            dims_2d_match = re.search(
                r'\b(\d{1,3}(?:\.\d+)?)\s*[xX×]\s*(\d{1,3}(?:\.\d+)?)\b(?!\s*[xX×])',  # Negative lookahead to avoid matching 3D
                text
            )
            if dims_2d_match:
                w, l = float(dims_2d_match.group(1)), float(dims_2d_match.group(2))
                # Only accept if these look like room dimensions (> 5 ft, < 200 ft)
                if all(5 <= x <= 200 for x in (w, l)):
                    data['width_ft'] = w
                    data['length_ft'] = l
                    # Height will default later if not found
        
        # Architectural format: 40'-0" x 25'-6"
        if 'width_ft' not in data:
            arch_match = re.search(
                r"(\d+)['\s]*[-–]?\s*(\d+)\s*[\"″]\s*[xX×]\s*(\d+)['\s]*[-–]?\s*(\d+)\s*[\"″]",
                text
            )
            if arch_match:
                data['width_ft'] = int(arch_match.group(1)) + int(arch_match.group(2)) / 12
                data['length_ft'] = int(arch_match.group(3)) + int(arch_match.group(4)) / 12
        
        # Individual dimensions with labels
        if 'width_ft' not in data:
            width_match = re.search(r'width[:\s]+(\d+\.?\d*)\s*(ft|feet|m|\')?', text, re.IGNORECASE)
            if width_match:
                val = float(width_match.group(1))
                unit = width_match.group(2) or 'ft'
                data['width_ft'] = val / 0.3048 if unit == 'm' else val
        
        if 'length_ft' not in data:
            length_match = re.search(r'length[:\s]+(\d+\.?\d*)\s*(ft|feet|m|\')?', text, re.IGNORECASE)
            if length_match:
                val = float(length_match.group(1))
                unit = length_match.group(2) or 'ft'
                data['length_ft'] = val / 0.3048 if unit == 'm' else val
        
        # Ceiling Height
        if 'height_ft' not in data:
            height_patterns = [
                r'(\d{1,2}(?:\.\d+)?)\s*(\'|ft|foot|feet)\s*(ceiling|h\b|high|height)',
                r'ceiling[:\s]+(\d{1,2}(?:\.\d+)?)\s*(\'|ft|m)?',
                r'height[:\s]+(\d{1,2}(?:\.\d+)?)\s*(\'|ft|m)?',
            ]
            for pattern in height_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    # Check for meters
                    if match.lastindex >= 2 and match.group(2) == 'm':
                        val = val / 0.3048
                    if 6 <= val <= 30:  # Reasonable ceiling height
                        data['height_ft'] = val
                        break
        
        # =====================================================================
        # HVAC PARAMETERS
        # =====================================================================
        
        # CFM / Airflow
        cfm_patterns = [
            r'(\d{2,5})\s*cfm',           # "1500 CFM" or "1500cfm"
            r'cfm[:\s]+(\d{2,5})',         # "CFM: 250" or "CFM 250"
            r'airflow[:\s]+(\d{2,5})',     # "Airflow: 1500"
            r'flow[:\s]+(\d{2,5})\s*(?:cfm)?',  # "Flow: 1500"
        ]
        for pattern in cfm_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 50 <= val <= 50000:  # Reasonable CFM range
                    data['airflow_cfm'] = val
                    break
        
        # Supply Temperature
        # LaTeX: $T=285.93 \text{ K}$ or $285.93 \text{ K}$
        temp_k_match = re.search(r'T\s*=\s*(\d+\.?\d*)\s*(?:\\text\{\s*K\s*\}|K)', text)
        if not temp_k_match:
            temp_k_match = re.search(r'\$(\d{3}\.?\d*)\s*\\text\{\s*K\s*\}\$', text)
        
        if temp_k_match:
            temp_k = float(temp_k_match.group(1))
            if 250 <= temp_k <= 350:  # Reasonable HVAC range
                # Convert K to F
                data['supply_temp_f'] = (temp_k - 273.15) * 9/5 + 32
        
        # Standard temp patterns (ordered from most specific to most general)
        if 'supply_temp_f' not in data:
            temp_patterns = [
                r'(\d{2})\s*°?\s*[fF]\s*(supply|sat)',            # "55F supply" or "55°F supply"
                r'supply\s*(?:air\s*)?(?:temp(?:erature)?)?[:\s]+(\d{2})\s*°?\s*[fF]?',  # "supply temp: 55F"
                r'sat[:\s]+(\d{2})\s*°?\s*[fF]?',                 # "sat: 55"
                r'(?:supply|air)\s+(?:at|@)\s*(\d{2})\s*°?\s*[fF]',  # "supply at 55°F" or "air at 52°F"
                r'(\d{2})\s*°\s*[fF]',                            # General "52°F" anywhere
            ]
            for pattern in temp_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    if 40 <= val <= 80:  # Reasonable supply temp
                        data['supply_temp_f'] = val
                        break
        
        # Velocity (m/s or fpm)
        # LaTeX: $y=-3.28 \text{ m/s}$
        vel_ms_match = re.search(r'[xyz]\s*=\s*(-?\d+\.?\d*)(?=.*(?:m/s|\\text\{\s*m/s))', text)
        if vel_ms_match:
            velocities = re.findall(r'[xyz]\s*=\s*(-?\d+\.?\d*)(?=.*(?:m/s|\\text\{\s*m/s))', text)
            non_zero = [abs(float(v)) for v in velocities if abs(float(v)) > 0.1]
            if non_zero:
                # Convert m/s to fpm
                data['velocity_fpm'] = max(non_zero) * 196.85
        
        # Standard velocity patterns
        if 'velocity_fpm' not in data:
            vel_fpm_match = re.search(r'(\d+)\s*fpm', text, re.IGNORECASE)
            if vel_fpm_match:
                data['velocity_fpm'] = float(vel_fpm_match.group(1))
        
        # =====================================================================
        # HEAT LOADS
        # =====================================================================
        
        # Watts (LaTeX or standard)
        heat_matches = re.findall(r'\$?(\d+\.?\d*)\s*(?:\\text\{\s*W\s*\}|W(?:atts?)?)\$?', text, re.IGNORECASE)
        if heat_matches:
            # Filter reasonable values (10W - 10kW)
            valid_heat = [float(h) for h in heat_matches if 10 <= float(h) <= 10000]
            if valid_heat:
                data['heat_load_w'] = sum(valid_heat)
        
        # BTU/hr
        btu_match = re.search(r'(\d+(?:,\d{3})*)\s*btu', text, re.IGNORECASE)
        if btu_match and 'heat_load_w' not in data:
            btu = float(btu_match.group(1).replace(',', ''))
            data['heat_load_w'] = btu / 3.412  # Convert to Watts
        
        # =====================================================================
        # DIFFUSER INFO
        # =====================================================================
        
        # Diffuser dimensions
        diff_match = re.search(r'diffuser[:\s]+(\d+)\s*[xX×]\s*(\d+)', text, re.IGNORECASE)
        if diff_match:
            data['diffuser_width_in'] = float(diff_match.group(1))
            data['diffuser_height_in'] = float(diff_match.group(2))
        
        # Vent count
        vent_match = re.search(r'(\d+)\s*(?:vents?|diffusers?|registers?)', text, re.IGNORECASE)
        if vent_match:
            data['vent_count'] = int(vent_match.group(1))
        
        return data
    
    # =========================================================================
    # UI Payload Builder
    # =========================================================================
    
    def _build_ui_payload(self, scanned_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """
        Wraps values in {value, source, status} for the Frontend Staging Area.
        """
        payload = {
            "success": True,
            "raw_text": raw_text[:5000] if raw_text else "",  # Truncate for display
            "fields": {}
        }
        
        fields = payload["fields"]
        
        # =====================================================================
        # PROJECT INFO
        # =====================================================================
        
        if 'project_name' in scanned_data:
            fields['project_name'] = UIField(
                value=scanned_data['project_name'],
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.8
            ).to_dict()
        else:
            fields['project_name'] = UIField(
                value=None,
                source=FieldSource.MISSING.value,
                status=FieldStatus.REQUIRED.value,
                confidence=0.0
            ).to_dict()
        
        if 'room_name' in scanned_data:
            fields['room_name'] = UIField(
                value=scanned_data['room_name'],
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.7
            ).to_dict()
        else:
            fields['room_name'] = UIField(
                value="Main Room",
                source=FieldSource.DEFAULT.value,
                status=FieldStatus.AUTO_FILLED.value,
                confidence=1.0
            ).to_dict()
        
        # =====================================================================
        # GEOMETRY
        # =====================================================================
        
        # Room Dimensions (Width x Length)
        if 'width_ft' in scanned_data and 'length_ft' in scanned_data:
            fields['room_width'] = UIField(
                value=round(scanned_data['width_ft'], 1),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.85
            ).to_dict()
            fields['room_length'] = UIField(
                value=round(scanned_data['length_ft'], 1),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.85
            ).to_dict()
        else:
            fields['room_width'] = UIField(
                value=None,
                source=FieldSource.MISSING.value,
                status=FieldStatus.REQUIRED.value,
                confidence=0.0
            ).to_dict()
            fields['room_length'] = UIField(
                value=None,
                source=FieldSource.MISSING.value,
                status=FieldStatus.REQUIRED.value,
                confidence=0.0
            ).to_dict()
        
        # Ceiling Height
        if 'height_ft' in scanned_data:
            fields['room_height'] = UIField(
                value=round(scanned_data['height_ft'], 1),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.8
            ).to_dict()
        else:
            fields['room_height'] = UIField(
                value=self.defaults['ceiling_height_ft'],
                source=FieldSource.DEFAULT.value,
                status=FieldStatus.AUTO_FILLED.value,
                confidence=1.0
            ).to_dict()
        
        # =====================================================================
        # HVAC PARAMETERS
        # =====================================================================
        
        # Airflow CFM
        if 'airflow_cfm' in scanned_data:
            fields['inlet_cfm'] = UIField(
                value=round(scanned_data['airflow_cfm'], 0),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.9
            ).to_dict()
        else:
            fields['inlet_cfm'] = UIField(
                value=None,
                source=FieldSource.MISSING.value,
                status=FieldStatus.REQUIRED.value,
                confidence=0.0
            ).to_dict()
        
        # Supply Temperature
        if 'supply_temp_f' in scanned_data:
            fields['supply_temp'] = UIField(
                value=round(scanned_data['supply_temp_f'], 1),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.85
            ).to_dict()
        else:
            fields['supply_temp'] = UIField(
                value=self.industry_standards['supply_temp_f'],
                source=FieldSource.INDUSTRY_STANDARD.value,
                status=FieldStatus.AUTO_FILLED.value,
                confidence=1.0
            ).to_dict()
        
        # Velocity (optional - can be calculated from CFM)
        if 'velocity_fpm' in scanned_data:
            fields['inlet_velocity'] = UIField(
                value=round(scanned_data['velocity_fpm'], 0),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.75
            ).to_dict()
        
        # =====================================================================
        # DIFFUSER INFO
        # =====================================================================
        
        if 'diffuser_width_in' in scanned_data:
            fields['diffuser_width'] = UIField(
                value=scanned_data['diffuser_width_in'],
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.8
            ).to_dict()
            fields['diffuser_height'] = UIField(
                value=scanned_data.get('diffuser_height_in', scanned_data['diffuser_width_in']),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.8
            ).to_dict()
        else:
            fields['diffuser_width'] = UIField(
                value=24,
                source=FieldSource.DEFAULT.value,
                status=FieldStatus.AUTO_FILLED.value,
                confidence=1.0
            ).to_dict()
            fields['diffuser_height'] = UIField(
                value=24,
                source=FieldSource.DEFAULT.value,
                status=FieldStatus.AUTO_FILLED.value,
                confidence=1.0
            ).to_dict()
        
        if 'vent_count' in scanned_data:
            fields['vent_count'] = UIField(
                value=scanned_data['vent_count'],
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.9
            ).to_dict()
        else:
            fields['vent_count'] = UIField(
                value=1,
                source=FieldSource.DEFAULT.value,
                status=FieldStatus.AUTO_FILLED.value,
                confidence=1.0
            ).to_dict()
        
        # =====================================================================
        # HEAT LOADS
        # =====================================================================
        
        if 'heat_load_w' in scanned_data:
            # Convert W to BTU/hr for display
            btu = scanned_data['heat_load_w'] * 3.412
            fields['heat_load'] = UIField(
                value=round(btu, 0),
                source=FieldSource.EXTRACTED.value,
                status=FieldStatus.REVIEW.value,
                confidence=0.7
            ).to_dict()
        else:
            fields['heat_load'] = UIField(
                value=0,
                source=FieldSource.DEFAULT.value,
                status=FieldStatus.AUTO_FILLED.value,
                confidence=1.0
            ).to_dict()
        
        # =====================================================================
        # SUMMARY STATS
        # =====================================================================
        
        extracted_count = sum(1 for f in fields.values() if f['source'] == 'extracted')
        required_count = sum(1 for f in fields.values() if f['status'] == 'required')
        review_count = sum(1 for f in fields.values() if f['status'] == 'review')
        
        payload["summary"] = {
            "total_fields": len(fields),
            "extracted": extracted_count,
            "required": required_count,
            "review": review_count,
            "ready_to_submit": required_count == 0
        }
        
        return payload


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    parser = HVACDocumentParser()
    
    # Test with sample text
    test_text = """
    PROJECT: Executive Office 404
    ROOM SIZE: 14x16
    CEILING: 10 ft drop ceiling
    SUPPLY: 1 VAV box @ 350 CFM
    Supply Temp: 55F
    """
    
    result = parser.parse_text_content(test_text)
    print(json.dumps(result, indent=2))
