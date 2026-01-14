"""
FastAPI Backend for Circuit Synthesis.

Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from config import CORS_ORIGINS, NUM_FREQ, FREQ_MIN, FREQ_MAX
from model_utils import get_model, list_available_models, switch_model, get_current_model_id

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Circuit Synthesis API",
    description="Generate RLC circuits from impedance curves using AI",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
    """Request to generate circuit from impedance."""
    magnitude: List[float]  # log10(|Z|) for each frequency
    phase: List[float]      # Phase in radians
    tau: Optional[float] = 0.5
    num_candidates: Optional[int] = 10
    model_id: Optional[str] = None  # Model to use (defaults to current)


class ComponentResponse(BaseModel):
    """A single circuit component."""
    type: str
    type_id: int
    node_a: int
    node_b: int
    value: float
    formatted_value: str


class ImpedanceResponse(BaseModel):
    """Impedance curve response."""
    magnitude: List[float]
    phase: List[float]
    frequencies: List[float]


class ErrorResponse(BaseModel):
    """Error metrics."""
    magnitude: float
    phase: float
    total: float


class CandidateResponse(BaseModel):
    """A circuit candidate."""
    components: List[ComponentResponse]
    impedance: ImpedanceResponse
    error: ErrorResponse


class GenerateResponse(BaseModel):
    """Response from circuit generation."""
    success: bool
    message: Optional[str] = None
    best: Optional[CandidateResponse] = None
    candidates: List[CandidateResponse] = []
    num_candidates: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    current_model: Optional[str] = None
    current_model_name: Optional[str] = None


class ConfigResponse(BaseModel):
    """Current configuration."""
    num_freq: int
    freq_min: float
    freq_max: float
    frequencies: List[float]


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    model = get_model()
    models_list = list_available_models()
    current = next((m for m in models_list if m['is_current']), None)
    return HealthResponse(
        status="ok",
        model_loaded=model.model is not None,
        device=str(model.device),
        current_model=get_current_model_id(),
        current_model_name=current['name'] if current else None,
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration (frequencies, etc.)."""
    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_FREQ)
    return ConfigResponse(
        num_freq=NUM_FREQ,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        frequencies=freqs.tolist(),
    )


@app.get("/models")
async def get_models():
    """List available models."""
    return {
        "models": list_available_models(),
        "current": get_current_model_id(),
    }


class SwitchModelRequest(BaseModel):
    """Request to switch model."""
    model_id: str


@app.post("/models/switch")
async def switch_model_endpoint(request: SwitchModelRequest):
    """Switch to a different model."""
    result = switch_model(request.model_id)
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Switch failed'))
    return result


@app.post("/generate", response_model=GenerateResponse)
async def generate_circuit(request: GenerateRequest):
    """
    Generate circuit from impedance curve.

    The impedance should be provided as:
    - magnitude: log10(|Z|) values for each frequency point
    - phase: Phase in radians for each frequency point

    Returns the best matching circuit and all candidates.
    """
    # Validate input
    if len(request.magnitude) != NUM_FREQ:
        raise HTTPException(
            status_code=400,
            detail=f"magnitude must have {NUM_FREQ} points, got {len(request.magnitude)}"
        )
    if len(request.phase) != NUM_FREQ:
        raise HTTPException(
            status_code=400,
            detail=f"phase must have {NUM_FREQ} points, got {len(request.phase)}"
        )

    # Create impedance array
    impedance = np.array([request.magnitude, request.phase])

    # Get model and generate (use specified model_id if provided)
    model = get_model(request.model_id)
    result = model.generate(
        impedance,
        tau=request.tau,
        num_candidates=request.num_candidates,
    )

    if not result['success']:
        return GenerateResponse(
            success=False,
            message=result.get('message', 'Generation failed'),
        )

    # Format response
    def format_candidate(cand):
        return CandidateResponse(
            components=[
                ComponentResponse(
                    type=c['type'],
                    type_id=c['type_id'],
                    node_a=c['node_a'],
                    node_b=c['node_b'],
                    value=c['value'],
                    formatted_value=model.format_component_value(c),
                )
                for c in cand['components']
            ],
            impedance=ImpedanceResponse(**cand['impedance']),
            error=ErrorResponse(**cand['error']),
        )

    return GenerateResponse(
        success=True,
        best=format_candidate(result['best']),
        candidates=[format_candidate(c) for c in result['candidates']],
        num_candidates=result['num_candidates'],
    )


@app.post("/compute-impedance")
async def compute_impedance_endpoint(components: List[dict]):
    """
    Compute impedance for given components.

    Useful for testing or computing Z(f) for manually specified circuits.
    """
    model = get_model()

    # Convert to expected format
    formatted_components = []
    for c in components:
        formatted_components.append({
            'type': c['type'],
            'type_id': {'R': 1, 'L': 2, 'C': 3}[c['type']],
            'node_a': c['node_a'],
            'node_b': c['node_b'],
            'value': c['value'],
        })

    z = model.compute_impedance_for_components(formatted_components)

    if z is None:
        raise HTTPException(status_code=400, detail="Could not compute impedance")

    return ImpedanceResponse(**z)


# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("Loading model...")
    get_model()
    print("Model ready!")


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
