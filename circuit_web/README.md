# Circuit Synthesis Web Application

Web interface for the Circuit Synthesis AI model. Generate RLC circuits from impedance curves Z(f).

## Architecture

```
circuit_web/
├── backend/          # FastAPI Python backend
│   ├── config.py     # ⚠️ MODIFY HERE to change model
│   ├── main.py       # API endpoints
│   ├── model_utils.py
│   └── requirements.txt
└── frontend/         # Next.js React frontend
    ├── src/
    │   ├── app/      # Pages
    │   ├── components/
    │   └── lib/      # API client
    └── package.json
```

## Quick Start

### 1. Start Backend

```bash
cd backend

# Install dependencies (first time)
pip install -r requirements.txt

# Run server
python main.py
# Or: uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

### 2. Start Frontend

```bash
cd frontend

# Install dependencies (first time)
npm install

# Run development server
npm run dev
```

Frontend runs at: http://localhost:3000

## Changing the Model

To use a different model checkpoint, edit `backend/config.py`:

```python
# Path to model checkpoint
MODEL_CHECKPOINT = Path("/path/to/your/model.pt")

# Model architecture (must match checkpoint)
MODEL_CONFIG = {
    "latent_dim": 256,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/config` | GET | Get frequency config |
| `/generate` | POST | Generate circuit from impedance |
| `/compute-impedance` | POST | Compute Z(f) for components |

### Generate Circuit Request

```json
POST /generate
{
  "magnitude": [2.5, 2.6, ...],  // log10(|Z|), 100 points
  "phase": [0.1, 0.15, ...],     // radians, 100 points
  "tau": 0.5,                    // temperature (optional)
  "num_candidates": 10           // Best-of-N (optional)
}
```

### Response

```json
{
  "success": true,
  "best": {
    "components": [
      {"type": "R", "node_a": 0, "node_b": 1, "value": 1000, "formatted_value": "1.00kΩ"},
      {"type": "C", "node_a": 1, "node_b": 0, "value": 1e-7, "formatted_value": "100.00nF"}
    ],
    "impedance": {"magnitude": [...], "phase": [...], "frequencies": [...]},
    "error": {"magnitude": 0.05, "phase": 0.02, "total": 0.052}
  },
  "candidates": [...],
  "num_candidates": 10
}
```

## Environment Variables

### Frontend (.env.local)

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (config.py)

- `MODEL_CHECKPOINT`: Path to model file
- `MODEL_CONFIG`: Model architecture parameters
- `API_HOST`, `API_PORT`: Server binding
- `CORS_ORIGINS`: Allowed frontend origins

## Development

### Adding new sample curves

Edit `frontend/src/components/ImpedanceInput.tsx`:

```typescript
const SAMPLE_CURVES = [
  {
    name: 'My Custom Circuit',
    generate: (freqs: number[]) => {
      // Return array of {magnitude, phase} for each frequency
      return freqs.map(f => ({
        magnitude: Math.log10(...),
        phase: Math.atan2(...)
      }));
    }
  },
  // ...
];
```

### Customizing the frontend

- `src/components/ImpedanceChart.tsx` - Chart visualization
- `src/components/CircuitDisplay.tsx` - Circuit topology display
- `src/app/page.tsx` - Main page layout

## Troubleshooting

### Backend won't start

1. Check Python version (3.8+)
2. Install dependencies: `pip install -r requirements.txt`
3. Check model path in `config.py`

### Frontend can't connect to backend

1. Ensure backend is running on port 8000
2. Check CORS settings in `backend/config.py`
3. Check `NEXT_PUBLIC_API_URL` in frontend

### Model loading fails

1. Verify checkpoint path exists
2. Check `MODEL_CONFIG` matches trained model architecture
3. Ensure circuit_transformer module is accessible
