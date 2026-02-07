# Modal APIs - Coqui TTS & Whisper STT

Production-ready Text-to-Speech and Speech-to-Text APIs hosted on Modal, designed for reuse across multiple projects.

## Features

### Coqui TTS API (✅ Implemented)
- **Built-in Speaker TTS**: 58 pre-trained voices from Coqui XTTS v2
- **Voice Cloning**: Clone any voice from 6-30 seconds of reference audio
- **Smart Caching**: Stale-while-revalidate pattern for speaker metadata
- **Fast Cold Starts**: 8s model load from Volume (vs 60s download), 3s with memory snapshotting
- **Auto-Chunking**: Handles long text with automatic sentence-boundary chunking
- **Audio Stitching**: Seamless crossfade between chunks with normalization

### Whisper STT API (🚧 Coming Soon)
- OpenAI Whisper for speech-to-text
- Multiple model sizes (tiny, base, small, medium, large)
- Multi-language support

## Quick Start

### 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv pip install -e ".[dev]"

# Install Modal
uv pip install modal
```

### 2. Set Up Modal

```bash
# Authenticate with Modal (interactive)
python3 -m modal setup

# Set Modal token (from your Modal dashboard)
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

### 3. Download Models to Modal Volume

```bash
# Download Coqui XTTS v2 model (~1.8GB) to Modal Volume
# This is a one-time setup (takes 5-10 minutes)
modal run coqui_service/download_models.py
```

### 4. Test Locally

```bash
# Serve dev endpoint (creates temporary URL)
modal serve coqui_service/main.py

# You'll see a URL like: https://yourname--coqui-apis-dev.modal.run
# Test endpoints:
# - GET /health
# - GET /speakers
# - POST /tts
# - POST /voice-clone
```

### 5. Deploy to Production

```bash
# Manual deployment
modal deploy coqui_service/main.py

# Or push to GitHub (auto-deploys via GitHub Actions)
git add .
git commit -m "Deploy Coqui TTS API"
git push origin main
```

## API Endpoints

### Base URL
- **Dev**: `https://yourname--coqui-apis-dev.modal.run` (temporary, via `modal serve`)
- **Prod**: `https://yourname--coqui-apis.modal.run` (stable, via `modal deploy`)

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "speakers_available": 58,
  "version": "0.1.0"
}
```

### GET /speakers
List available built-in speakers with caching.

**Query Parameters:**
- `refresh` (bool, optional): Force cache refresh (default: false)

**Response:**
```json
{
  "speakers": ["Aaron Dreschner", "Abrahan Mack", ...],
  "count": 58,
  "last_updated": "2026-02-07T12:00:00",
  "cache_age_days": 2
}
```

### POST /tts
Synthesize speech using built-in speaker.

**Request Body:**
```json
{
  "text": "Hello world, this is a test.",
  "speaker_id": "Aaron Dreschner",
  "language": "en",
  "speed": 1.0
}
```

**Response:** WAV audio file

**Headers:**
- `X-Sample-Rate`: 22050
- `X-Duration-Sec`: 3.45
- `X-Engine`: coqui_xtts
- `X-Speaker`: Aaron Dreschner
- `X-Chunks`: 1

### POST /voice-clone
Synthesize speech using voice cloning.

**Request (multipart/form-data):**
- `text` (string): Text to synthesize
- `language` (string): Language code (default: "en")
- `reference_audio` (file): Reference audio file (WAV, MP3, M4A)

**Response:** WAV audio file

## Development Workflow

### Dev Cycle (Git-Based)

1. **Create feature branch**:
   ```bash
   git checkout -b feature/add-french-support
   ```

2. **Iterate with dev endpoint**:
   ```bash
   modal serve coqui_service/main.py
   # Test changes at temporary dev URL
   # Make code changes, endpoint auto-reloads
   ```

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Add French language support"
   git push origin feature/add-french-support
   ```

4. **Merge to main → Auto-deploy**:
   - Create PR, review, merge to `main`
   - GitHub Actions automatically deploys to production

### GitHub Actions Setup

Add these secrets to your GitHub repository:
1. Go to repo **Settings** → **Secrets and variables** → **Actions**
2. Add `MODAL_TOKEN_ID` (from Modal dashboard)
3. Add `MODAL_TOKEN_SECRET` (from Modal dashboard)

## Project Structure

```
modal_apis/
├── coqui_service/          # Coqui TTS API
│   ├── main.py             # Modal app entry point
│   ├── engine.py           # TTS engine wrapper
│   ├── routes.py           # FastAPI routes
│   ├── models.py           # Pydantic models
│   ├── download_models.py  # Model download script
│   └── utils/
│       ├── chunker.py      # Text chunking
│       ├── audio.py        # Audio processing
│       └── speaker_cache.py # Speaker caching
├── whisper_service/        # Future: Whisper STT API
├── .github/
│   └── workflows/
│       └── deploy.yml      # Auto-deployment
├── execution_docs/         # Planning & execution docs
├── pyproject.toml          # Dependencies (uv)
└── README.md
```

## Supported Languages

Coqui XTTS v2 supports 17 languages:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Polish (pl)
- Turkish (tr)
- Russian (ru)
- Dutch (nl)
- Czech (cs)
- Arabic (ar)
- Chinese (zh-cn)
- Japanese (ja)
- Hungarian (hu)
- Korean (ko)
- Hindi (hi)

## Cost Estimation

Based on Modal pricing (as of 2026):

**TTS Request (3s inference on T4 GPU)**:
- GPU: $0.00051/s × 3s = $0.00153 per request
- **10,000 requests/month**: ~$15.30

**Speaker List (CPU-only, cached)**:
- CPU: $0.0001/s × 0.1s = $0.00001 per request
- **10,000 requests/month**: ~$0.10

**Volume Storage** (1.8GB model):
- $0.10/GB/month = $0.18/month

**Total for 10k requests**: ~$17/month

**Free Tier**: Modal provides $30/month credit, sufficient for ~15k requests.

## Troubleshooting

### Model not loading
```bash
# Re-download model to Volume
modal run coqui_service/download_models.py
```

### Speaker list empty
```bash
# Force refresh speaker cache
curl "https://yourname--coqui-apis.modal.run/speakers?refresh=true"
```

### Cold start too slow
- Memory snapshotting is enabled by default (8s → 3s)
- Consider `keep_warm=1` for production (costs ~$0.50/hour idle)

### GitHub Actions failing
- Verify `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` secrets are set
- Check Modal token has not expired

## License

MIT

## Contributing

1. Fork the repo
2. Create feature branch
3. Test with `modal serve`
4. Submit PR to `main`
