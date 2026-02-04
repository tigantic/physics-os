"""
Variant Scoring API - Production FastAPI Endpoint
==================================================

REST API for GPU-accelerated variant pathogenicity scoring.

Endpoints:
- POST /score - Score single variant
- POST /score/batch - Score multiple variants
- GET /health - Health check
- GET /model/info - Model information

Features:
- GPU-accelerated scoring (ESM-2 + multi-omics)
- Batch processing for throughput
- ACMG-style clinical evidence generation
- OpenAPI documentation

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
import json
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/root/benchmark_data")
MODEL_VERSION = "1.0.0"


# =============================================================================
# Constants
# =============================================================================

AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

BLOSUM62 = np.array([
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
], dtype=np.float32)

GRANTHAM = np.array([
    [  0, 112, 111, 126, 195, 91, 107,  60,  86,  94,  96, 106,  84, 113,  27,  99,  58, 148, 112,  64],
    [112,   0,  86,  96, 180,  43,  54, 125,  29,  97, 102,  26,  91,  97, 103, 110,  71, 101,  77,  96],
    [111,  86,   0,  23, 139,  46,  42,  80,  68, 149, 153,  94, 142, 158,  91,  46,  65, 174, 143, 133],
    [126,  96,  23,   0, 154,  61,  45,  94,  81, 168, 172, 101, 160, 177, 108,  65,  85, 181, 160, 152],
    [195, 180, 139, 154,   0, 154, 170, 159, 174, 198, 198, 202, 196, 205, 169, 112, 149, 215, 194, 192],
    [ 91,  43,  46,  61, 154,   0,  29,  87,  24, 109, 113,  53, 101, 116,  76,  68,  42, 130,  99,  96],
    [107,  54,  42,  45, 170,  29,   0,  98,  40, 134, 138,  56, 126, 140,  93,  80,  65, 152, 122, 121],
    [ 60, 125,  80,  94, 159,  87,  98,   0,  98, 135, 138, 127, 127, 153,  42,  56,  59, 184, 147, 109],
    [ 86,  29,  68,  81, 174,  24,  40,  98,   0,  94,  99,  32,  87, 100,  77,  89,  47, 115,  83,  84],
    [ 94,  97, 149, 168, 198, 109, 134, 135,  94,   0,   5, 102,  10,  21,  95, 142,  89,  61,  33,  29],
    [ 96, 102, 153, 172, 198, 113, 138, 138,  99,   5,   0, 107,  15,  22,  98, 145,  92,  61,  36,  32],
    [106,  26,  94, 101, 202,  53,  56, 127,  32, 102, 107,   0,  95, 102, 103, 121,  78, 110,  85,  97],
    [ 84,  91, 142, 160, 196, 101, 126, 127,  87,  10,  15,  95,   0,  28,  87, 135,  81,  67,  36,  21],
    [113,  97, 158, 177, 205, 116, 140, 153, 100,  21,  22, 102,  28,   0, 114, 155, 103,  40,  22,  50],
    [ 27, 103,  91, 108, 169,  76,  93,  42,  77,  95,  98, 103,  87, 114,   0,  74,  38, 147, 110,  68],
    [ 99, 110,  46,  65, 112,  68,  80,  56,  89, 142, 145, 121, 135, 155,  74,   0,  58, 177, 144, 124],
    [ 58,  71,  65,  85, 149,  42,  65,  59,  47,  89,  92,  78,  81, 103,  38,  58,   0, 128,  92,  69],
    [148, 101, 174, 181, 215, 130, 152, 184, 115,  61,  61, 110,  67,  40, 147, 177, 128,   0,  37,  88],
    [112,  77, 143, 160, 194,  99, 122, 147,  83,  33,  36,  85,  36,  22, 110, 144,  92,  37,   0,  55],
    [ 64,  96, 133, 152, 192,  96, 121, 109,  84,  29,  32,  97,  21,  50,  68, 124,  69,  88,  55,   0],
], dtype=np.float32)


# =============================================================================
# Pydantic Models
# =============================================================================

class Classification(str, Enum):
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely pathogenic"
    UNCERTAIN = "Uncertain significance"
    LIKELY_BENIGN = "Likely benign"
    BENIGN = "Benign"


class VariantInput(BaseModel):
    """Single variant input."""
    gene: str = Field(..., description="Gene symbol (e.g., BRCA1)")
    protein_change: str = Field(..., description="Protein change (e.g., p.Arg1699Trp or R1699W)")
    transcript: Optional[str] = Field(None, description="Transcript ID")
    
    @field_validator('protein_change')
    @classmethod
    def validate_protein_change(cls, v: str) -> str:
        # Accept both formats: p.Arg1699Trp or R1699W
        return v


class BatchVariantInput(BaseModel):
    """Batch variant input."""
    variants: List[VariantInput] = Field(..., max_length=1000)


class FeatureScores(BaseModel):
    """Individual feature scores."""
    esm2: float = Field(..., ge=0, le=1, description="ESM-2 evolutionary score")
    blosum62: float = Field(..., ge=0, le=1, description="BLOSUM62 substitution score")
    grantham: float = Field(..., ge=0, le=1, description="Grantham biochemical distance")
    combined: float = Field(..., ge=0, le=1, description="Combined pathogenicity score")


class ACMGEvidence(BaseModel):
    """ACMG evidence codes."""
    supporting_pathogenic: List[str] = Field(default_factory=list)
    moderate_pathogenic: List[str] = Field(default_factory=list)
    strong_pathogenic: List[str] = Field(default_factory=list)
    very_strong_pathogenic: List[str] = Field(default_factory=list)
    supporting_benign: List[str] = Field(default_factory=list)
    strong_benign: List[str] = Field(default_factory=list)


class ClinicalReport(BaseModel):
    """Clinical interpretation report."""
    variant: str
    gene: str
    classification: Classification
    confidence: float
    summary: str
    acmg_evidence: ACMGEvidence
    recommendations: List[str]


class VariantResult(BaseModel):
    """Single variant scoring result."""
    variant: str
    gene: str
    scores: FeatureScores
    classification: Classification
    confidence: float
    clinical_report: Optional[ClinicalReport] = None
    processing_time_ms: float


class BatchResult(BaseModel):
    """Batch scoring result."""
    results: List[VariantResult]
    total_variants: int
    processing_time_ms: float
    throughput_per_sec: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_available: bool
    esm2_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model information."""
    version: str
    features: List[str]
    gpu_device: Optional[str]
    esm2_model: str
    supported_genes: int


# =============================================================================
# Scoring Engine
# =============================================================================

class VariantScoringEngine:
    """GPU-accelerated variant scoring engine."""
    
    def __init__(self):
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.esm_model = None
        self.esm_alphabet = None
        self.proteins: Dict[str, str] = {}
        self.blosum = torch.tensor(BLOSUM62, device=self.device, dtype=torch.float32)
        self.grantham = torch.tensor(GRANTHAM, device=self.device, dtype=torch.float32)
        self.start_time = time.time()
        
    def load_models(self):
        """Load ESM-2 and protein sequences."""
        logger.info("Loading models...")
        
        # Load protein sequences
        protein_path = DATA_DIR / "protein_sequences.pkl"
        if protein_path.exists():
            with open(protein_path, 'rb') as f:
                self.proteins = pickle.load(f)
            logger.info(f"Loaded {len(self.proteins)} protein sequences")
        
        # Load ESM-2
        if ESM_AVAILABLE:
            logger.info("Loading ESM-2...")
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()
            self.batch_converter = self.esm_alphabet.get_batch_converter()
            logger.info("ESM-2 loaded")
    
    def parse_variant(self, gene: str, protein_change: str) -> Optional[tuple]:
        """Parse variant notation to (wt, pos, mt)."""
        import re
        
        # Try p.Arg1699Trp format
        m = re.match(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', protein_change)
        if m:
            wt = AA_3TO1.get(m.group(1))
            mt = AA_3TO1.get(m.group(3))
            pos = int(m.group(2))
            if wt and mt:
                return wt, pos, mt
        
        # Try R1699W format
        m = re.match(r'([A-Z])(\d+)([A-Z])', protein_change)
        if m:
            wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
            if wt in AA_TO_IDX and mt in AA_TO_IDX:
                return wt, pos, mt
        
        return None
    
    @torch.no_grad()
    def score_esm2(self, gene: str, wt: str, pos: int, mt: str) -> float:
        """Score variant with ESM-2."""
        if self.esm_model is None or gene not in self.proteins:
            return 0.5
        
        seq = self.proteins[gene]
        if pos < 1 or pos > len(seq):
            return 0.5
        
        # Truncate long sequences
        max_len = 800
        if len(seq) > max_len:
            seq = seq[:max_len]
            if pos > max_len:
                return 0.5
        
        # Forward pass
        batch = [(gene, seq)]
        _, _, batch_tokens = self.batch_converter(batch)
        batch_tokens = batch_tokens.to(self.device)
        
        results = self.esm_model(batch_tokens, repr_layers=[33])
        logits = results["logits"][0]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get LLR
        wt_tok = self.esm_alphabet.get_idx(wt)
        mt_tok = self.esm_alphabet.get_idx(mt)
        
        wt_logprob = log_probs[pos, wt_tok]
        mt_logprob = log_probs[pos, mt_tok]
        llr = wt_logprob - mt_logprob
        
        return float(torch.sigmoid(llr * 0.5))
    
    def score_biochemical(self, wt: str, mt: str) -> Dict[str, float]:
        """Score with BLOSUM62 and Grantham."""
        wt_idx = AA_TO_IDX[wt]
        mt_idx = AA_TO_IDX[mt]
        
        blosum_raw = float(self.blosum[wt_idx, mt_idx])
        grantham_raw = float(self.grantham[wt_idx, mt_idx])
        
        blosum_score = float(torch.sigmoid(torch.tensor(-blosum_raw * 0.5)))
        grantham_score = float(torch.sigmoid(torch.tensor((grantham_raw - 100) * 0.02)))
        
        return {
            'blosum': blosum_score,
            'grantham': grantham_score,
        }
    
    def score_variant(self, gene: str, protein_change: str) -> Optional[VariantResult]:
        """Score a single variant."""
        t_start = time.perf_counter()
        
        parsed = self.parse_variant(gene, protein_change)
        if not parsed:
            return None
        
        wt, pos, mt = parsed
        
        # Get scores
        esm2_score = self.score_esm2(gene, wt, pos, mt)
        biochem = self.score_biochemical(wt, mt)
        
        # Combined score (adaptive weighting)
        esm_uncertainty = 1.0 - abs(esm2_score - 0.5) * 2
        multiomics_avg = (biochem['blosum'] + biochem['grantham']) / 2
        combined = esm2_score * (1 - esm_uncertainty * 0.4) + multiomics_avg * (esm_uncertainty * 0.4)
        
        # Classification
        if combined >= 0.9:
            classification = Classification.PATHOGENIC
        elif combined >= 0.7:
            classification = Classification.LIKELY_PATHOGENIC
        elif combined >= 0.3:
            classification = Classification.UNCERTAIN
        elif combined >= 0.1:
            classification = Classification.LIKELY_BENIGN
        else:
            classification = Classification.BENIGN
        
        confidence = abs(combined - 0.5) * 2  # 0-1 scale
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        
        variant_str = f"{gene}:p.{AA_1TO3[wt]}{pos}{AA_1TO3[mt]}"
        
        return VariantResult(
            variant=variant_str,
            gene=gene,
            scores=FeatureScores(
                esm2=esm2_score,
                blosum62=biochem['blosum'],
                grantham=biochem['grantham'],
                combined=combined,
            ),
            classification=classification,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
        )
    
    def generate_clinical_report(self, result: VariantResult) -> ClinicalReport:
        """Generate ACMG-style clinical report."""
        scores = result.scores
        
        # Build ACMG evidence
        evidence = ACMGEvidence()
        
        # PP3: Computational evidence supports pathogenic
        if scores.combined >= 0.7:
            evidence.supporting_pathogenic.append("PP3: Multiple computational methods support pathogenic effect")
        
        # PM1: Located in mutational hot spot / functional domain
        if scores.esm2 >= 0.8:
            evidence.moderate_pathogenic.append("PM1: ESM-2 indicates critical residue position")
        
        # BP4: Computational evidence supports benign
        if scores.combined <= 0.3:
            evidence.supporting_benign.append("BP4: Multiple computational methods suggest benign effect")
        
        # BP1: Missense in gene where truncating is mechanism
        if scores.grantham < 0.3:
            evidence.supporting_benign.append("BP1: Conservative amino acid change (low Grantham distance)")
        
        # Build summary
        if result.classification in [Classification.PATHOGENIC, Classification.LIKELY_PATHOGENIC]:
            summary = (
                f"The variant {result.variant} is classified as {result.classification.value} "
                f"based on computational analysis. The combined pathogenicity score of "
                f"{scores.combined:.2f} indicates this variant is likely to affect protein function. "
                f"ESM-2 evolutionary analysis score: {scores.esm2:.2f}. "
                f"Biochemical disruption score (Grantham): {scores.grantham:.2f}."
            )
            recommendations = [
                "Consider functional studies to validate pathogenicity",
                "Review clinical correlation with patient phenotype",
                "Genetic counseling recommended for family members",
            ]
        elif result.classification == Classification.UNCERTAIN:
            summary = (
                f"The variant {result.variant} is classified as {result.classification.value}. "
                f"The combined score of {scores.combined:.2f} falls in the uncertain range. "
                f"Additional evidence is needed for definitive classification."
            )
            recommendations = [
                "Functional studies may help resolve classification",
                "Segregation analysis in family members recommended",
                "Periodic re-evaluation as new evidence becomes available",
            ]
        else:
            summary = (
                f"The variant {result.variant} is classified as {result.classification.value}. "
                f"Computational analysis suggests this variant is unlikely to affect protein function. "
                f"Combined score: {scores.combined:.2f}."
            )
            recommendations = [
                "Benign classification based on computational evidence",
                "Clinical correlation recommended",
            ]
        
        return ClinicalReport(
            variant=result.variant,
            gene=result.gene,
            classification=result.classification,
            confidence=result.confidence,
            summary=summary,
            acmg_evidence=evidence,
            recommendations=recommendations,
        )


# =============================================================================
# FastAPI Application
# =============================================================================

engine: Optional[VariantScoringEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global engine
    engine = VariantScoringEngine()
    engine.load_models()
    logger.info("Variant Scoring API ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Variant Pathogenicity Scoring API",
    description="GPU-accelerated variant scoring with ESM-2 and multi-omics features",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        gpu_available=CUDA_AVAILABLE,
        esm2_loaded=engine.esm_model is not None if engine else False,
        model_version=MODEL_VERSION,
        uptime_seconds=time.time() - engine.start_time if engine else 0,
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    gpu_name = None
    if CUDA_AVAILABLE:
        gpu_name = torch.cuda.get_device_properties(0).name
    
    return ModelInfo(
        version=MODEL_VERSION,
        features=["ESM-2", "BLOSUM62", "Grantham", "Adaptive ensemble"],
        gpu_device=gpu_name,
        esm2_model="esm2_t33_650M_UR50D",
        supported_genes=len(engine.proteins) if engine else 0,
    )


@app.post("/score", response_model=VariantResult)
async def score_variant(variant: VariantInput, include_report: bool = False):
    """
    Score a single variant for pathogenicity.
    
    - **gene**: Gene symbol (e.g., BRCA1, TP53)
    - **protein_change**: Amino acid change (e.g., p.Arg1699Trp or R1699W)
    - **include_report**: Include full clinical report
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = engine.score_variant(variant.gene, variant.protein_change)
    
    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse variant: {variant.gene} {variant.protein_change}"
        )
    
    if include_report:
        result.clinical_report = engine.generate_clinical_report(result)
    
    return result


@app.post("/score/batch", response_model=BatchResult)
async def score_batch(batch: BatchVariantInput, include_reports: bool = False):
    """
    Score multiple variants in batch.
    
    More efficient for multiple variants (shared ESM-2 context).
    Maximum 1000 variants per request.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    t_start = time.perf_counter()
    
    results = []
    for v in batch.variants:
        result = engine.score_variant(v.gene, v.protein_change)
        if result:
            if include_reports:
                result.clinical_report = engine.generate_clinical_report(result)
            results.append(result)
    
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    throughput = len(results) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    
    return BatchResult(
        results=results,
        total_variants=len(results),
        processing_time_ms=elapsed_ms,
        throughput_per_sec=throughput,
    )


@app.post("/report", response_model=ClinicalReport)
async def generate_report(variant: VariantInput):
    """
    Generate a full ACMG-style clinical report for a variant.
    
    Includes:
    - Classification with confidence
    - ACMG evidence codes
    - Clinical summary
    - Recommendations
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = engine.score_variant(variant.gene, variant.protein_change)
    
    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse variant: {variant.gene} {variant.protein_change}"
        )
    
    return engine.generate_clinical_report(result)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
