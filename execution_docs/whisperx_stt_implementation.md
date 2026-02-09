# WhisperX STT Implementation on Modal

**Date**: 2026-02-09
**Status**: Implementation Complete - Ready for Testing
**Goal**: Build a self-hosted STT API with word-level timestamps for karaoke-style highlighting

---

## Requirements

### Model & Library
- **Model**: `large-v3-turbo` (6x faster than large-v3, negligible accuracy loss)
- **Library**: WhisperX (provides word-level timestamps via forced phoneme alignment)
- **Alignment Model**: Wav2Vec2 (for frame-accurate word boundaries)

### Infrastructure
- **Platform**: Modal
- **GPU**: A10G (24GB VRAM - holds both Whisper + Alignment models)
- **Storage**: Modal Volume for caching:
  - WhisperX model weights
  - Wav2Vec2 alignment model weights
- **Loading Strategy**: `@modal.enter()` hook in Modal Class

### Output Format
- Full JSON object from WhisperX
- Word-level segments with `start` and `end` timestamps
- Frame-accurate timing suitable for TikTok/Reels karaoke effects

---

## Architecture Plan

### File Structure (mirroring Coqui setup)
```
whisper_service/
├── main.py              # Modal app + ASGI endpoint
├── engine.py            # WhisperX engine wrapper (Class-based with @enter hook)
├── routes.py            # FastAPI endpoints
├── models.py            # Pydantic request/response models
├── download_models.py   # Pre-download models to Volume
└── utils/
    └── audio_utils.py   # Audio processing helpers
```

### Key Components

#### 1. Modal Volume Setup
```python
# Volume for model caching
volume = modal.Volume.from_name("whisperx-models-v1", create_if_missing=True)

# Models to cache:
# - WhisperX large-v3-turbo (~3GB)
# - Wav2Vec2 alignment model (~1.2GB)
# - VAD model (optional, ~300MB)
```

#### 2. WhisperX Engine Class (engine.py)
```python
@app.cls(
    image=image,
    gpu="A10G",  # 24GB VRAM
    volumes={"/models": volume},
    timeout=600,  # 10 min max
    enable_memory_snapshot=True,  # Fast cold starts
    scaledown_window=120,  # Keep alive 2min
)
class WhisperXEngine:
    @modal.enter()
    def load_models(self):
        """Load WhisperX + alignment models from Volume on container startup."""
        # Load WhisperX model (large-v3-turbo)
        # Load alignment model (Wav2Vec2)
        # Load VAD model (optional)

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        """Transcribe audio and return word-level timestamps."""
        # Whisper transcription
        # Forced alignment (Wav2Vec2)
        # Return full JSON with word-level segments
```

#### 3. FastAPI Endpoints (routes.py)
- `POST /transcribe` - Upload audio, get word-level transcript
- `GET /health` - Health check
- `GET /docs` - OpenAPI docs

#### 4. Model Download Script (download_models.py)
```python
@app.function(volumes={"/models": volume}, timeout=1200)
def download_whisperx_models():
    """Download WhisperX large-v3-turbo + alignment models to Volume."""
    # Download Whisper model
    # Download Wav2Vec2 alignment model
    # Commit to volume
```

---

## Implementation Steps

### Phase 1: Setup & Dependencies ✅
- [x] Create `whisper_service/` package structure
- [x] Define Docker image with dependencies:
  - whisperx==3.1.1
  - torch + torchaudio >=2.1.0
  - faster-whisper >=0.10.0
  - transformers >=4.36.0
  - pyannote.audio >=3.1.0 (VAD)
  - fastapi[standard] >=0.108.0
  - python-multipart
  - ffmpeg (system dependency)

### Phase 2: Model Download ✅
- [x] Create `download_models.py`
- [x] Implement download function for:
  - WhisperX large-v3-turbo (~3GB)
  - Wav2Vec2 alignment model (~1.2GB)
  - VAD model (~300MB)
- [ ] Test: `modal run whisper_service/download_models.py` (NEXT STEP)

### Phase 3: Engine Implementation ✅
- [x] Create `engine.py` with `WhisperXEngine` class
- [x] Implement `load_models()` method for @modal.enter() hook
- [x] Implement `transcribe()` method:
  - Audio preprocessing with VAD
  - Whisper transcription with auto-language detection
  - Forced alignment (Wav2Vec2) for word-level timestamps
  - Return word-level segments with confidence scores
- [x] Support for 99+ languages (SUPPORTED_LANGUAGES dict)
- [x] Handle audio format conversion (accept WAV, MP3, M4A, etc.)

### Phase 4: API Endpoints ✅
- [x] Create `models.py` (Pydantic schemas):
  - WordSegment (word, start, end, score)
  - Segment (text, start, end, words[])
  - TranscribeRequest/Response
  - HealthResponse
  - LanguagesResponse
- [x] Create `routes.py`:
  - `POST /transcribe` - File upload + transcription
  - `GET /health` - Health check
  - `GET /languages` - List supported languages
  - `GET /docs` - Auto-generated OpenAPI docs
- [x] Implement error handling and logging

### Phase 5: Modal Integration ✅
- [x] Create `main.py`:
  - Volume setup (whisperx-models-v1)
  - Image definition with all dependencies
  - Modal ASGI function (GPU, memory snapshot, volume mount)
  - FastAPI app mounting
  - @web_app.on_event("startup") for model pre-loading
- [ ] Test locally: `modal serve whisper_service/main.py` (NEXT STEP)

### Phase 6: Testing & Validation
- [ ] Test with sample audio files
- [ ] Verify word-level timestamps accuracy
- [ ] Check GPU memory usage (should fit in 24GB)
- [ ] Benchmark cold start time (with memory snapshot)
- [ ] Benchmark transcription latency

### Phase 7: Documentation & Deployment
- [ ] Update `docs/testing.md` with STT testing procedures
- [ ] Update `CLAUDE.md` with WhisperX service info
- [ ] Add GitHub Actions workflow (mirror Coqui deployment)
- [ ] Deploy to production: `modal deploy whisper_service/main.py`

---

## Technical Details

### WhisperX vs Standard Whisper
| Feature | Standard Whisper | WhisperX |
|---------|-----------------|----------|
| Timestamps | Segment-level (2-5s) | Word-level (<100ms accuracy) |
| Alignment | Soft attention | Forced phoneme alignment (Wav2Vec2) |
| Use Case | General transcription | Karaoke, subtitles, precise timing |

### GPU Memory Budget (A10G = 24GB)
- WhisperX large-v3-turbo: ~6GB VRAM
- Wav2Vec2 alignment: ~2GB VRAM
- Audio buffer + inference: ~2GB VRAM
- **Total**: ~10GB VRAM (50% headroom for safety)

### Expected Performance
- **Cold start** (with memory snapshot): ~5-8s
- **Transcription**: ~0.3x realtime (1 min audio → 18s transcription)
- **Alignment**: +2-5s overhead
- **Total latency**: ~20-25s for 1 min audio

---

## Open Questions

1. **Audio Format Support**: Should we support all formats (WAV, MP3, M4A, FLAC) or restrict to WAV?
   - Recommendation: Accept all, convert to WAV internally using ffmpeg

2. **VAD (Voice Activity Detection)**: Should we use VAD to skip silence?
   - Recommendation: Yes, improves accuracy and speed for long audio with pauses

3. **Batch Processing**: Should we support batch transcription (multiple files)?
   - Recommendation: Start with single-file, add batch later if needed

4. **Language Detection**: Auto-detect language or require user input?
   - Recommendation: Auto-detect by default, allow override via API parameter

5. **Diarization**: Should we add speaker diarization (who said what)?
   - Recommendation: Out of scope for v1, can add later with pyannote.audio

---

## Implementation Summary

**✅ Completed (2026-02-09)**

All core components have been implemented:

1. **Package Structure**: `whisper_service/` with proper module organization
2. **Models** (`models.py`): Pydantic schemas for request/response validation
3. **Audio Utils** (`utils/audio_utils.py`): FFmpeg-based audio conversion to 16kHz mono WAV
4. **Engine** (`engine.py`): WhisperXEngine class with:
   - `load_models()` for @modal.enter() hook
   - `transcribe()` with word-level timestamps
   - Support for 99+ languages
   - Auto-language detection
5. **Routes** (`routes.py`): Three FastAPI endpoints:
   - `POST /transcribe` - File upload + transcription
   - `GET /health` - Service health check
   - `GET /languages` - List of 99+ supported languages
6. **Model Download** (`download_models.py`): Script to pre-download models to Volume
7. **Main App** (`main.py`): Modal ASGI app with:
   - A10G GPU configuration
   - Volume mounting at `/models`
   - Memory snapshot enabled
   - Container keep-alive (120s)

---

## Status

- [x] Requirements gathered
- [x] Architecture designed
- [x] Execution plan created
- [x] User approval received
- [x] Core implementation complete
- [ ] **Model download test (NEXT)**
- [ ] **Local deployment test (NEXT)**
- [ ] Documentation updates

---

## Next Steps

1. Download models to Modal Volume: `modal run whisper_service/download_models.py`
2. Test local deployment: `modal serve whisper_service/main.py`
3. Test transcription with sample audio files
4. Update `CLAUDE.md` and `docs/testing.md`
5. Add GitHub Actions workflow for CI/CD

