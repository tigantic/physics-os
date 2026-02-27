"""
Image Extractor
===============

Extracts data from blueprint images (PNG, JPG, TIFF).
Uses OCR for text recognition and image analysis for scale detection.
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


class ImageExtractor(BaseExtractor):
    """
    Extract HVAC data from blueprint images.
    
    Features:
    - OCR text extraction via Tesseract
    - Scale bar detection and measurement
    - Dimension line recognition
    - Room boundary detection
    """
    
    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]
    SUPPORTED_MIME_TYPES = [
        "image/png", "image/jpeg", "image/tiff", "image/bmp"
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for required dependencies."""
        self._has_pil = False
        self._has_tesseract = False
        self._has_cv2 = False
        
        try:
            from PIL import Image
            self._has_pil = True
        except ImportError:
            pass
        
        try:
            import pytesseract
            self._has_tesseract = True
        except ImportError:
            pass
        
        try:
            import cv2
            self._has_cv2 = True
        except ImportError:
            pass
    
    def extract(self, file_path) -> ExtractionResult:
        """Extract data from an image file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            return self._create_empty_result(
                file_path.name, "image",
                f"File not found: {file_path}"
            )
        
        with open(file_path, "rb") as f:
            data = f.read()
        
        return self.extract_from_bytes(data, file_path.name)
    
    def extract_from_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """Extract data from image bytes."""
        if not self._has_pil:
            return self._create_empty_result(
                filename, "image",
                "Pillow not installed. Run: pip install Pillow"
            )
        
        from PIL import Image
        
        file_hash = self._compute_hash(data)
        fields: Dict[str, ExtractedField] = {}
        warnings: List[str] = []
        errors: List[str] = []
        raw_text = ""
        
        try:
            # Open image
            img = Image.open(io.BytesIO(data))
            
            # Store image info
            fields["image_width_px"] = ExtractedField(
                name="image_width_px",
                value=img.width,
                confidence=ExtractionConfidence.HIGH,
            )
            fields["image_height_px"] = ExtractedField(
                name="image_height_px",
                value=img.height,
                confidence=ExtractionConfidence.HIGH,
            )
            
            # Create thumbnail for preview
            thumb = img.copy()
            thumb.thumbnail((400, 400))
            preview_buffer = io.BytesIO()
            thumb.save(preview_buffer, format="PNG")
            preview_image = preview_buffer.getvalue()
            
            # OCR if available
            if self._has_tesseract:
                import pytesseract
                
                # Preprocess for better OCR
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Extract text
                raw_text = pytesseract.image_to_string(img)
                
                # Also try to get structured data
                try:
                    ocr_data = pytesseract.image_to_data(
                        img, output_type=pytesseract.Output.DICT
                    )
                    fields.update(self._process_ocr_data(ocr_data))
                except Exception as e:
                    warnings.append(f"Structured OCR failed: {e}")
            else:
                warnings.append("Tesseract not available - no text extraction")
            
            # Detect unit system from extracted text
            unit_system, unit_confidence = detect_unit_system(raw_text)
            
            # Extract structured fields from text
            if raw_text:
                fields.update(self._extract_dimensions(raw_text))
                fields.update(self._extract_hvac_data(raw_text))
                fields.update(self._extract_project_info(raw_text))
                fields.update(self._extract_scale_from_text(raw_text))
            
            # Try computer vision analysis
            if self._has_cv2:
                cv_fields = self._analyze_blueprint_cv(data)
                fields.update(cv_fields)
            
            return ExtractionResult(
                success=True,
                file_name=filename,
                file_type="image",
                file_hash=file_hash,
                extracted_at=datetime.now(),
                detected_unit_system=unit_system,
                unit_confidence=unit_confidence,
                fields=fields,
                warnings=warnings,
                errors=errors,
                raw_text=raw_text,
                preview_image=preview_image,
            )
            
        except Exception as e:
            return self._create_empty_result(
                filename, "image",
                f"Image extraction failed: {str(e)}"
            )
    
    def _process_ocr_data(self, ocr_data: dict) -> Dict[str, ExtractedField]:
        """Process structured OCR data to find measurements and dimensions."""
        fields = {}
        
        # Combine words with high confidence
        words = []
        word_info = []  # Store word with position info
        for i, word in enumerate(ocr_data.get("text", [])):
            conf = ocr_data.get("conf", [0])[i]
            if conf > 50 and word.strip():  # 50% confidence threshold
                words.append(word.strip())
                word_info.append({
                    "text": word.strip(),
                    "conf": conf,
                    "left": ocr_data.get("left", [0])[i],
                    "top": ocr_data.get("top", [0])[i],
                })
        
        text = " ".join(words)
        
        # Look for architectural dimension patterns: 40'-0", 25'-6", etc.
        ft_in_pattern = r"(\d+)[\'\']\s*[-–]?\s*(\d+)[\"″\"]?"
        ft_in_matches = re.findall(ft_in_pattern, text)
        
        if ft_in_matches:
            # Convert to decimal feet
            dims = []
            for feet, inches in ft_in_matches:
                try:
                    decimal_ft = int(feet) + int(inches)/12
                    if decimal_ft >= 1:  # Filter noise
                        dims.append(decimal_ft)
                except ValueError:
                    pass
            
            if dims:
                # Sort and get largest (likely room dimensions)
                dims_sorted = sorted(set(dims), reverse=True)
                # Filter reasonable room dimensions (5-200 ft)
                reasonable = [d for d in dims_sorted if 5 <= d <= 200]
                
                fields["all_dimensions_ft"] = ExtractedField(
                    name="all_dimensions_ft",
                    value=reasonable[:10],
                    confidence=ExtractionConfidence.LOW,
                    original_text=str(ft_in_matches[:5]),
                )
        
        # Also look for simple numbers that could be dimensions (10', 25', etc.)
        simple_ft_pattern = r"(\d+(?:\.\d+)?)\s*[\'']"
        simple_matches = re.findall(simple_ft_pattern, text)
        
        if simple_matches:
            simple_dims = []
            for m in simple_matches:
                try:
                    val = float(m)
                    if 5 <= val <= 200:  # Reasonable dimension
                        simple_dims.append(val)
                except ValueError:
                    pass
            
            if simple_dims and "all_dimensions_ft" not in fields:
                fields["all_dimensions_ft"] = ExtractedField(
                    name="all_dimensions_ft",
                    value=sorted(set(simple_dims), reverse=True)[:10],
                    confidence=ExtractionConfidence.LOW,
                    original_text=str(simple_matches[:5]),
                )
        
        return fields
    
    def _extract_scale_from_text(self, text: str) -> Dict[str, ExtractedField]:
        """Extract scale information from OCR text."""
        fields = {}
        
        # Common scale notations
        scale_patterns = [
            r"scale[:\s]+1[:\s/](\d+)",
            r"1/(\d+)\s*[\"']\s*=\s*1\s*['\"]",  # 1/4" = 1'
            r"(\d+)\s*[\"']\s*=\s*(\d+)\s*['\"]",  # 1" = 10'
            r"1\s*:\s*(\d+)",  # 1:100
            r"not\s*to\s*scale",  # NTS indicator
        ]
        
        for pattern in scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "not" in pattern:
                    fields["scale_warning"] = ExtractedField(
                        name="scale_warning",
                        value="NOT TO SCALE",
                        confidence=ExtractionConfidence.HIGH,
                        original_text=match.group(0),
                    )
                else:
                    fields["detected_scale"] = ExtractedField(
                        name="detected_scale",
                        value=match.group(0),
                        confidence=ExtractionConfidence.MEDIUM,
                        original_text=match.group(0),
                    )
                break
        
        return fields
    
    def _analyze_blueprint_cv(self, image_data: bytes) -> Dict[str, ExtractedField]:
        """Use computer vision to analyze blueprint structure."""
        if not self._has_cv2:
            return {}
        
        import cv2
        import numpy as np
        
        fields = {}
        
        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect lines (for dimension lines and room boundaries)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, threshold=100,
                minLineLength=50, maxLineGap=10
            )
            
            if lines is not None:
                # Count horizontal and vertical lines
                h_lines = 0
                v_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    if angle < 10 or angle > 170:
                        h_lines += 1
                    elif 80 < angle < 100:
                        v_lines += 1
                
                fields["detected_h_lines"] = ExtractedField(
                    name="detected_h_lines",
                    value=h_lines,
                    confidence=ExtractionConfidence.LOW,
                )
                fields["detected_v_lines"] = ExtractedField(
                    name="detected_v_lines",
                    value=v_lines,
                    confidence=ExtractionConfidence.LOW,
                )
            
            # Detect rectangles (potential rooms)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            large_rects = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10000:  # Significant area
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) == 4:
                        large_rects += 1
            
            if large_rects > 0:
                fields["detected_rooms_cv"] = ExtractedField(
                    name="detected_rooms_cv",
                    value=large_rects,
                    confidence=ExtractionConfidence.LOW,
                )
            
        except Exception:
            pass
        
        return fields
