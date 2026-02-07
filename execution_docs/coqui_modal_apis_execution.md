# Coqui Modal APIs - Execution Document

**Date**: 2026-02-07
**Status**: ✅ Implementation Complete - Ready for Testing
**Planning Doc**: [coqui_modal_apis_planning.md](./coqui_modal_apis_planning.md)

## Implementation Summary

All core components have been implemented:
- ✅ Project structure with `uv` package manager
- ✅ Pydantic models for API requests/responses
- ✅ Text chunking utilities (retrofitted from story_reels)
- ✅ Audio stitching and processing utilities
- ✅ Speaker metadata caching with stale-while-revalidate
- ✅ TTS engine wrapper with Modal Volume loading
- ✅ FastAPI routes (TTS, speakers, voice-clone, health)
- ✅ Modal ASGI app with GPU, memory snapshotting
- ✅ Model download script
- ✅ GitHub Actions auto-deployment workflow
- ✅ Comprehensive documentation (README, SETUP_INSTRUCTIONS)

**Next Steps**: Run setup commands and test all endpoints (see SETUP_INSTRUCTIONS.md)

---

## Project Structure

```
modal_apis/
├── .github/
│   └── workflows/
│       └── deploy.yml                 # GitHub Actions auto-deployment
├── coqui_service/
│   ├── __init__.py
│   ├── main.py                        # Modal app entry point (@modal.asgi_app)
│   ├── engine.py                      # TTS engine wrapper (load/inference)
│   ├── routes.py                      # FastAPI route handlers
│   ├── models.py                      # Pydantic request/response models
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── chunker.py                 # Text chunking (from story_reels)
│   │   ├── stitching.py               # Audio stitching (from story_reels)
│   │   └── speaker_cache.py           # Speaker metadata caching
│   └── tests/
│       ├── test_tts.py
│       ├── test_speakers.py
│       └── test_clone.py
├── whisper_service/                   # Future: STT service
│   └── main.py
├── shared/                            # Shared utilities across services
│   ├── __init__.py
│   └── audio_processing.py
├── execution_docs/
│   ├── coqui_modal_apis_planning.md
│   └── coqui_modal_apis_execution.md
├── pyproject.toml                     # uv package manager config
├── .gitignore
├── .python-version                    # Python 3.10
└── README.md
```

---

## Development Workflow (Git-Based)

### Dev Cycle

1. **Branch**: `git checkout -b feature/add-feature`
2. **Iterate**: `modal serve coqui_service/main.py`
   - Creates temporary dev endpoint (e.g., `https://yourname--coqui-dev.modal.run`)
   - Test changes without breaking production API
3. **Test**: Run tests locally, verify dev endpoint
4. **Commit**: `git add . && git commit -m "Add feature"`
5. **Push**: `git push origin feature/add-feature`
6. **PR → Merge**: Merge to `main` triggers GitHub Actions auto-deploy
7. **Production**: Modal auto-deploys to stable URL (e.g., `https://yourname--coqui.modal.run`)

### GitHub Actions Auto-Deployment

On push to `main`, `.github/workflows/deploy.yml` automatically:
- Installs Modal CLI
- Deploys `coqui_service/main.py` to production
- Runs health checks
- Posts deployment status

---

## Task Breakdown

### Phase 1: Project Setup ✅ / ⏳ / ❌

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Initialize project with `uv` | ⏳ | Create `pyproject.toml`, `.python-version` |
| 1.2 | Create directory structure | ⏳ | Folders for coqui_service, .github/workflows, etc. |
| 1.3 | Set up `.gitignore` | ⏳ | Ignore `__pycache__`, `.venv/`, Modal cache |
| 1.4 | Create `README.md` | ⏳ | Project overview, setup instructions |
| 1.5 | Install base dependencies | ⏳ | `uv pip install modal fastapi structlog` |

### Phase 2: Modal Infrastructure Setup

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | Create Modal Volume | ⏳ | `modal volume create coqui-models-v2` |
| 2.2 | Write model download script | ⏳ | `coqui_service/download_models.py` |
| 2.3 | Run model download to Volume | ⏳ | Download XTTS v2 (1.8GB) to Volume |
| 2.4 | Verify Volume contents | ⏳ | Check `/models/coqui/` has model files |
| 2.5 | Create Modal Image definition | ⏳ | Python 3.10, TTS, torch, ffmpeg |

### Phase 3: Core TTS Engine

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | Create `coqui_service/engine.py` | ⏳ | TTS engine wrapper class |
| 3.2 | Implement model loading from Volume | ⏳ | Load from `/models/coqui/` |
| 3.3 | Implement speaker discovery | ⏳ | Use `tts.speakers` pattern from story_reels |
| 3.4 | Implement built-in speaker TTS | ⏳ | `tts.tts(text, speaker, language)` |
| 3.5 | Implement voice cloning TTS | ⏳ | `tts.tts(text, speaker_wav, language)` |
| 3.6 | Add error handling | ⏳ | Invalid speaker, OOM, model load failures |

### Phase 4: Utilities (Copy from story_reels)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | Copy `chunker.py` from story_reels | ⏳ | `tts_v2/pipeline/chunker.py` → `utils/chunker.py` |
| 4.2 | Copy `stitching.py` from story_reels | ⏳ | `tts_v2/pipeline/stitching.py` → `utils/stitching.py` |
| 4.3 | Copy `postprocess.py` from story_reels | ⏳ | Audio normalization, WAV wrapping |
| 4.4 | Adapt utilities for Modal | ⏳ | Remove orchestrator dependencies |
| 4.5 | Create `speaker_cache.py` | ⏳ | Stale-while-revalidate logic for speaker metadata |

### Phase 5: Pydantic Models

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | Define `TTSRequest` model | ⏳ | text, speaker_id, language, speed |
| 5.2 | Define `SpeakersResponse` model | ⏳ | speakers list, count, last_updated |
| 5.3 | Define `VoiceCloneRequest` model | ⏳ | text, reference_audio, language |
| 5.4 | Define error response models | ⏳ | Standard error format with codes |
| 5.5 | Add validation rules | ⏳ | Text length limits, language enum, etc. |

### Phase 6: FastAPI Routes

| # | Task | Status | Notes |
|---|------|--------|-------|
| 6.1 | Create `coqui_service/routes.py` | ⏳ | Define all route handlers |
| 6.2 | Implement `POST /tts` | ⏳ | Built-in speaker synthesis |
| 6.3 | Implement `GET /speakers` | ⏳ | List speakers with caching |
| 6.4 | Implement `POST /voice-clone` | ⏳ | Voice cloning synthesis |
| 6.5 | Implement `GET /health` | ⏳ | Health check endpoint |
| 6.6 | Add error handlers | ⏳ | 400, 413, 500 error responses |

### Phase 7: Modal App Entry Point

| # | Task | Status | Notes |
|---|------|--------|-------|
| 7.1 | Create `coqui_service/main.py` | ⏳ | Modal App definition |
| 7.2 | Define Modal Image | ⏳ | debian_slim + TTS + torch + ffmpeg |
| 7.3 | Define Modal Volume mount | ⏳ | `/models` → `coqui-models-v2` |
| 7.4 | Implement `@modal.asgi_app()` | ⏳ | FastAPI app factory |
| 7.5 | Configure GPU (T4) | ⏳ | `gpu=modal.gpu.T4()` |
| 7.6 | Configure keep-warm | ⏳ | `keep_warm=1` for low latency |
| 7.7 | Configure concurrency | ⏳ | `@modal.concurrent(max_inputs=10)` |
| 7.8 | Add memory snapshotting | ⏳ | Optional: `enable_memory_snapshot=True` |

### Phase 8: Speaker Metadata Caching

| # | Task | Status | Notes |
|---|------|--------|-------|
| 8.1 | Implement cache read from Volume | ⏳ | Load `speaker_metadata.json` |
| 8.2 | Implement cache age check | ⏳ | Check if <10 days old |
| 8.3 | Implement async refresh trigger | ⏳ | If stale, trigger background refresh |
| 8.4 | Implement cache write to Volume | ⏳ | Write new JSON + `volume.commit()` |
| 8.5 | Add `?refresh=true` query param | ⏳ | Force synchronous refresh |

### Phase 9: Testing

| # | Task | Status | Notes |
|---|------|--------|-------|
| 9.1 | Write unit tests for engine | ⏳ | Test speaker discovery, TTS, cloning |
| 9.2 | Write unit tests for utils | ⏳ | Test chunker, stitching, cache logic |
| 9.3 | Write integration tests | ⏳ | Test full API flows |
| 9.4 | Test with `modal serve` (dev) | ⏳ | Local testing before deployment |
| 9.5 | Verify speaker list caching | ⏳ | Test cache hit/miss/refresh |
| 9.6 | Test error scenarios | ⏳ | Invalid speaker, missing audio, etc. |

### Phase 10: GitHub Actions CI/CD

| # | Task | Status | Notes |
|---|------|--------|-------|
| 10.1 | Create `.github/workflows/deploy.yml` | ⏳ | Auto-deploy on push to main |
| 10.2 | Configure Modal secrets | ⏳ | Add `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| 10.3 | Test deployment workflow | ⏳ | Push to test branch, verify deploy |
| 10.4 | Add health check to workflow | ⏳ | Verify `/health` returns 200 |
| 10.5 | Add deployment notifications | ⏳ | Optional: Slack/email on deploy |

### Phase 11: Documentation

| # | Task | Status | Notes |
|---|------|--------|-------|
| 11.1 | Write `README.md` | ⏳ | Setup, usage, API docs |
| 11.2 | Document API endpoints | ⏳ | Request/response examples |
| 11.3 | Add usage examples | ⏳ | Python client examples |
| 11.4 | Document dev workflow | ⏳ | Branch → serve → commit → deploy |
| 11.5 | Add troubleshooting guide | ⏳ | Common errors and fixes |

### Phase 12: Production Readiness

| # | Task | Status | Notes |
|---|------|--------|-------|
| 12.1 | Load test with 100 requests | ⏳ | Verify autoscaling works |
| 12.2 | Monitor cold start times | ⏳ | Should be <8s without snapshot, <3s with |
| 12.3 | Verify cost estimation | ⏳ | Check actual costs vs. estimate |
| 12.4 | Set up monitoring | ⏳ | Modal dashboard alerts |
| 12.5 | Document production URL | ⏳ | Add to README |
| 12.6 | Final smoke test | ⏳ | Test all 3 endpoints in production |

---

## Dependencies (uv)

```toml
[project]
name = "modal-apis"
version = "0.1.0"
description = "Modal-hosted TTS and STT APIs"
requires-python = ">=3.10"

dependencies = [
    "modal>=0.63.0",
    "fastapi[standard]>=0.108.0",
    "pydantic>=2.5.0",
    "TTS>=0.22.0",
    "torch>=2.1.0",
    "torchaudio>=2.1.0",
    "structlog>=23.1.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",  # For testing async endpoints
    "ruff>=0.1.0",
]
```

---

## Modal Secrets Setup

Before deploying, create Modal secrets:

```bash
# Set up Modal authentication
modal token new

# Secrets are stored in Modal dashboard
# Add to GitHub repo secrets:
# - MODAL_TOKEN_ID
# - MODAL_TOKEN_SECRET
```

---

## Commands

### Local Development
```bash
# Install dependencies
uv pip install -e ".[dev]"

# Serve dev endpoint (creates temporary URL)
modal serve coqui_service/main.py

# Run tests
pytest coqui_service/tests/

# Lint code
ruff check coqui_service/
```

### Deployment
```bash
# Manual deploy to production
modal deploy coqui_service/main.py

# GitHub Actions auto-deploys on push to main
git push origin main
```

### Modal Volume Management
```bash
# Create volume
modal volume create coqui-models-v2

# List volumes
modal volume list

# View volume contents
modal volume ls coqui-models-v2 /models/coqui/
```

---

## Questions for User

Before proceeding with implementation, I need to clarify:

1. **Modal Account**: Do you already have a Modal account set up? If not, we'll need to run `modal setup` first.

2. **GitHub Secrets**: Do you want me to add instructions for setting up Modal tokens in GitHub secrets, or will you handle that?

3. **Keep-Warm Strategy**: Should we start with `keep_warm=0` (scale to zero, cheaper) or `keep_warm=1` (always ready, faster)? We can adjust later.

4. **Memory Snapshotting**: Enable from the start (faster cold starts, slightly more complex) or add later as an optimization?

5. **Testing Strategy**: Should I write tests as I implement each component, or implement everything first then add tests?

6. **story_reels Access**: Can I copy the following files directly from story_reels?
   - `tts_v2/pipeline/chunker.py`
   - `tts_v2/pipeline/stitching.py`
   - `tts_v2/pipeline/postprocess.py`
   - `tts_v2/interfaces.py` (for data classes)

7. **Project Name**: The Modal app will be named `coqui-apis` by default. Should we use a different name?

**Ready to proceed?** Let me know your preferences and I'll start with Phase 1: Project Setup.
