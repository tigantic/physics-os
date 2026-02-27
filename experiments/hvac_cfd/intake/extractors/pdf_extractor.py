"""
PDF Extractor
=============

Extracts data from PDF blueprints and specification documents.
Uses PyMuPDF for text extraction and optional OCR for scanned documents.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io

try:
    from . import (
        BaseExtractor, ExtractionResult, ExtractedField, 
        ExtractionConfidence
    )
    from ..units import detect_unit_system, UnitSystem
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extractors import (
        BaseExtractor, ExtractionResult, ExtractedField, 
        ExtractionConfidence
    )
    from units import detect_unit_system, UnitSystem


class PDFExtractor(BaseExtractor):
    """
    Extract HVAC data from PDF documents.
    
    Handles:
    - Text-based PDFs (specification documents)
    - Scanned PDFs (blueprints) via OCR
    - Mixed documents with annotations
    """
    
    SUPPORTED_EXTENSIONS = [".pdf"]
    SUPPORTED_MIME_TYPES = ["application/pdf"]
    
    def __init__(self, *args, enable_ocr: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_ocr = enable_ocr
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for required dependencies."""
        self._has_pymupdf = False
        self._has_tesseract = False
        
        try:
            import fitz  # PyMuPDF
            self._has_pymupdf = True
        except ImportError:
            pass
        
        try:
            import pytesseract
            from PIL import Image
            self._has_tesseract = True
        except ImportError:
            pass
    
    def extract(self, file_path) -> ExtractionResult:
        """Extract data from a PDF file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            return self._create_empty_result(
                file_path.name, "pdf", 
                f"File not found: {file_path}"
            )
        
        with open(file_path, "rb") as f:
            data = f.read()
        
        return self.extract_from_bytes(data, file_path.name)
    
    def extract_from_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """Extract data from PDF bytes."""
        if not self._has_pymupdf:
            return self._create_empty_result(
                filename, "pdf",
                "PyMuPDF not installed. Run: pip install pymupdf"
            )
        
        import fitz
        
        file_hash = self._compute_hash(data)
        fields: Dict[str, ExtractedField] = {}
        warnings: List[str] = []
        errors: List[str] = []
        all_text = []
        preview_image = None
        
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            
            # Generate preview from first page
            if len(doc) > 0:
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                preview_image = pix.tobytes("png")
            
            # Extract text from all pages
            for page_num, page in enumerate(doc):
                # First try direct text extraction
                text = page.get_text()
                
                if text.strip():
                    all_text.append(f"[Page {page_num + 1}]\n{text}")
                elif self.enable_ocr and self._has_tesseract:
                    # Fall back to OCR for scanned pages
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        all_text.append(f"[Page {page_num + 1} - OCR]\n{ocr_text}")
                        warnings.append(f"Page {page_num + 1} required OCR")
                
                # Extract annotations/comments
                for annot in page.annots() or []:
                    if annot.info.get("content"):
                        all_text.append(f"[Annotation]\n{annot.info['content']}")
            
            doc.close()
            
            # Combine all text
            combined_text = "\n\n".join(all_text)
            
            # Detect unit system
            unit_system, unit_confidence = detect_unit_system(combined_text)
            
            # Extract structured fields
            fields.update(self._extract_dimensions(combined_text))
            fields.update(self._extract_hvac_data(combined_text))
            fields.update(self._extract_project_info(combined_text))
            fields.update(self._extract_blueprint_scale(combined_text))
            fields.update(self._extract_schedule_data(combined_text))
            
            return ExtractionResult(
                success=True,
                file_name=filename,
                file_type="pdf",
                file_hash=file_hash,
                extracted_at=datetime.now(),
                detected_unit_system=unit_system,
                unit_confidence=unit_confidence,
                fields=fields,
                warnings=warnings,
                errors=errors,
                raw_text=combined_text,
                preview_image=preview_image,
            )
            
        except Exception as e:
            return self._create_empty_result(
                filename, "pdf",
                f"PDF extraction failed: {str(e)}"
            )
    
    def _ocr_page(self, page) -> str:
        """Perform OCR on a PDF page."""
        if not self._has_tesseract:
            return ""
        
        import fitz
        import pytesseract
        from PIL import Image
        
        # Render page to image at 300 DPI
        mat = fitz.Matrix(300/72, 300/72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Run OCR
        try:
            text = pytesseract.image_to_string(img)
            return text
        except Exception:
            return ""
    
    def _extract_blueprint_scale(self, text: str) -> Dict[str, ExtractedField]:
        """Extract scale information from blueprint."""
        fields = {}
        
        # Scale patterns
        scale_patterns = [
            r"scale[:\s]+1[:\s/](\d+)",
            r"1\s*[:\"/]\s*(\d+)\s*[\"']?\s*=\s*1\s*[\"']?",  # 1/4" = 1'
            r"(\d+)\s*[\"']\s*=\s*(\d+)\s*['\"]",  # 1" = 10'
            r"1\s*:\s*(\d+)",  # 1:100
        ]
        
        for pattern in scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["blueprint_scale"] = ExtractedField(
                    name="blueprint_scale",
                    value=match.group(0),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                )
                break
        
        return fields
    
    def _extract_schedule_data(self, text: str) -> Dict[str, ExtractedField]:
        """Extract data from equipment schedules in PDF."""
        fields = {}
        
        # Diffuser schedule patterns
        diffuser_patterns = [
            # Look for tabular data with CFM values
            r"(?:diffuser|register|outlet)\s*(?:schedule|data).*?(\d+)\s*(?:CFM|cfm)",
            r"type[:\s]+([A-Z0-9\-]+).*?(\d+)\s*(?:CFM|cfm)",
        ]
        
        # AHU/RTU patterns
        ahu_patterns = [
            r"(?:AHU|RTU|air\s*handler)[:\s#]*(\d+).*?(\d+(?:,\d{3})?)\s*(?:CFM|cfm)",
            r"(?:supply|return)\s*(?:air)?[:\s]+(\d+(?:,\d{3})?)\s*(?:CFM|cfm)",
        ]
        
        for pattern in ahu_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the largest CFM value as total supply
                try:
                    cfm_values = [int(m[-1].replace(",", "")) for m in matches]
                    max_cfm = max(cfm_values)
                    fields["total_supply_cfm"] = ExtractedField(
                        name="total_supply_cfm",
                        value=max_cfm,
                        confidence=ExtractionConfidence.LOW,
                        original_text=str(matches),
                        unit="CFM",
                    )
                except (ValueError, IndexError):
                    pass
                break
        
        return fields
