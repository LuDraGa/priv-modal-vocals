# Coqui Modal APIs - Planning Document

**Date**: 2026-02-07
**Objective**: Build production-ready Coqui TTS, Speaker List, and Voice Cloning APIs on Modal with optimized cold-start performance

---

## 1. Core Architecture Decisions

### 1.1 Storage Strategy

**Model Storage**: `modal.Volume` (v2)
- **Why**: Models are large (1.8GB for XTTS v2), read-heavy, static assets
- **Why Not NFS**: NetworkFileSystem is single-region, causing variable latency in multi-region Modal compute (deprecated by Modal)
- **Volume v2 Benefits**: Distributed across all regions, higher throughput, improved random-access, true high-concurrency writes
- **Benefit**: 8s load from Volume vs 60s network download on every cold start (85% reduction)
- **Path**: `/models/coqui/xtts_v2/`
- **Setup**: One-time model download to volume during initialization

**Speaker Metadata**: `modal.Volume` (same volume as models)
- **Why**: Static JSON (<100KB), accessed frequently, rare writes
- **Benefit**: Millisecond reads, distributed across regions
- **Path**: `/models/coqui/speaker_metadata.json`
- **Pattern**: Stale-while-revalidate (10-day TTL)
- **Commit Performance**: Single small JSON write, no concurrency issues

**Generated Audio Cache** (future optimization): `modal.Volume` or S3
- TTS chunks are repeatable and cacheable
- Not implemented in v1 but architecture should support it

### 1.2 Modal Function Architecture

**Single Modal App with ASGI Router**:

Instead of separate `@app.function` decorators, use **`@modal.asgi_app()`** with FastAPI for unified routing:

```python
@app.function(
    image=image,
    gpu=modal.gpu.T4(),
    volumes={"/models": volume},
    keep_warm=1,  # Keep 1 container alive during active hours
    timeout=300,  # 5 min max per request
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    web_app = FastAPI()

    @web_app.post("/tts")
    async def tts_endpoint(...): ...

    @web_app.get("/speakers")
    async def speakers_endpoint(...): ...

    @web_app.post("/voice-clone")
    async def clone_endpoint(...): ...

    return web_app
```

**Why Single ASGI App?**
- **Shared GPU Pool**: All endpoints reuse same warmed-up containers
- **Shared Model Load**: XTTS loaded once, serves all endpoints
- **Cost Efficiency**: Speaker List uses same container (no extra charge for CPU-only function)
- **Simplified Deployment**: Single endpoint URL, one health check
- **Route-based Logic**: Use `if request.url.path == "/speakers"` to skip GPU operations for lightweight requests

---

## 2. API Specifications

### 2.1 TTS API

**Endpoint**: `POST /tts`

**Request Body**:
```json
{
  "text": "Hello world, this is a test.",
  "speaker_id": "Aaron Dreschner",
  "language": "en",
  "speed": 1.0,
  "output_format": "wav"
}
```

**Response**:
- `200 OK`: Binary audio (WAV format)
- `400 Bad Request`: Invalid speaker_id, unsupported language
- `500 Internal Error`: Model load failure, inference error

**Notes**:
- `speaker_id` must match exactly (case-sensitive)
- Supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
- `speed`: 0.7 - 1.3 (optional, default 1.0)
- Text chunked automatically if >200 chars (XTTS warning threshold = 250)

### 2.2 List Speakers API

**Endpoint**: `GET /speakers?refresh=false`

**Query Parameters**:
- `refresh`: boolean (default: false) - Force rebuild speaker list

**Response**:
```json
{
  "speakers": [
    "Aaron Dreschner",
    "Abrahan Mack",
    "Alexandra Hisakawa",
    ...
  ],
  "count": 58,
  "last_updated": "2026-02-07T10:15:30Z",
  "cache_age_days": 3
}
```

**Stale-While-Revalidate Logic**:
1. Check if `speaker_metadata.json` exists on Volume
2. If exists, check `last_modified` timestamp
   - If <10 days old: Return immediately (cache hit, <100ms)
   - If ≥10 days old: Return stale data + trigger async refresh
   - If missing: Synchronous rebuild (slow path, ~8s)
3. Refresh process (from story_reels pattern):
   ```python
   # DO NOT use SpeakerManager - too heavyweight
   # Use the loaded TTS instance's .speakers attribute
   if hasattr(tts, 'speakers') and tts.speakers:
       speakers = tts.speakers
       if isinstance(speakers, list):
           return speakers
       elif isinstance(speakers, dict):
           return list(speakers.keys())
   ```
   - Extract speaker names from `tts.speakers` (58 built-in speakers)
   - Write new JSON with metadata: `{"speakers": [...], "last_updated": "...", "count": 58}`
   - Call `volume.commit()` (fast for small JSON, <500ms)

**Why 10 days TTL?**
- Speakers are static (XTTS v2 has 58 fixed speakers)
- Only change with model version updates (rare)
- Balances freshness vs. unnecessary compute
- User can force refresh with `?refresh=true`

**Commit Performance Note**:
- Small JSON commits (<100KB) complete in <500ms
- Avoid concurrent commits (Modal recommends max 5 concurrent for small changes)
- Speaker metadata update is rare, no concurrency needed

### 2.3 Voice Clone API

**Endpoint**: `POST /voice-clone`

**Request Body** (multipart/form-data):
```
text: "This is the text to synthesize"
base_audio: <file upload> (optional - for base voice characteristics)
reference_audio: <file upload> (required - target voice)
language: "en"
output_format: "wav"
```

**Response**:
- `200 OK`: Binary audio (WAV format)
- `400 Bad Request`: Missing reference_audio, invalid audio format
- `413 Payload Too Large`: Audio file >10MB
- `500 Internal Error`: Inference failure

**Notes**:
- `reference_audio`: **6-30 seconds optimal** (XTTS v2 can clone from as little as 6s)
- For high-fidelity cloning: 20-30 seconds, clean audio, single speaker
- For fine-tuned cloning (future): 2 hours of audio cut into 3-10s clips
- `base_audio`: Optional. If provided, uses as style base + reference for timbre
- Audio formats: WAV, MP3, M4A (auto-converted to 22.05kHz mono WAV)
- Cloning quality depends heavily on reference audio clarity
- XTTS v2 supports **multiple reference files** for better quality (interpolation)

---

## 3. Model Loading Optimization

### 3.1 Cold Start Minimization

**Without Volume** (baseline):
```
Download model (60s) → Load to RAM (8s) → Inference (3s) = 71s total
```

**With Volume** (optimized):
```
Load from Volume to RAM (8s) → Inference (3s) = 11s total
```

**Savings**: 60s per cold start = 85% reduction

### 3.2 Volume Setup Process

**One-time initialization** (run once, persist forever):
```python
@app.function(volumes={"/models": volume})
def download_models():
    import os
    from TTS.api import TTS

    os.environ["TTS_HOME"] = "/models/coqui"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
    # Model downloads to /models/coqui automatically (1.8GB)
    volume.commit()  # Persist to Modal Volume (~5s for initial commit)
```

### 3.3 Modal Container Lifecycle Optimization

**Keep-Warm Strategy** (Modal 2025 parameters):
```python
@app.function(
    keep_warm=1,  # Keep 1 container always running
    min_containers=1,  # Floor on container count (doesn't scale to zero)
    scaledown_window=120,  # Keep idle containers alive for 2 min
)
```
- **keep_warm=1**: Ensures <500ms response time for first request
- **Cost**: ~$0.50/hour for idle GPU (vs $2-5 for cold start latency during peak)
- **When to use**: Set during active hours (8am-8pm), scale to 0 at night

**Memory Snapshotting** (advanced optimization):
```python
@app.function(
    enable_memory_snapshot=True,  # Snapshot after warmup
    experimental_options={"enable_gpu_snapshot": True}  # Snapshot GPU state
)
```
- **Benefit**: Reduces cold start from 8s (model load) to <3s (snapshot restore)
- **Tradeoff**: Slightly higher storage costs, but 60% faster cold starts
- **Use case**: High-traffic APIs where sub-3s cold starts matter

**Concurrency** (for ASGI app):
```python
@modal.concurrent(max_inputs=10)  # Process 10 requests concurrently
```
- **Why**: ASGI is async, can handle multiple requests per container
- **Speaker List**: Uses same container, no extra GPU needed (async I/O)
- **TTS/Clone**: GPU inference is sequential, but requests queued efficiently

---

## 4. Critical Implementation Details

### 4.1 Coqui-Specific Considerations

**Speaker Discovery** (for Speaker List API - from story_reels):
```python
# Use the already-loaded TTS instance (avoid extra loading)
# Reference: tts_v2/engines/coqui_xtts.py:149-161
def _discover_speakers(tts) -> List[str]:
    """Discover available speakers from loaded TTS model."""
    if hasattr(tts, 'speakers') and tts.speakers:
        speakers = tts.speakers
        if isinstance(speakers, list):
            return speakers
        elif isinstance(speakers, dict):
            return list(speakers.keys())
    return []
```

**Why this approach?**
- **No extra loading**: Reuses TTS instance already loaded for inference
- **Fast**: Just accesses `.speakers` attribute (no file I/O)
- **Proven**: Production-tested in story_reels
- **XTTS v2 speakers**: Returns list of 58 built-in speaker names (e.g., "Aaron Dreschner", "Ana Florence", etc.)
- **Can run on CLI**: `tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --list_speaker_idx`

**Built-in vs. Clone Mode** (TTS API):
```python
# Built-in speaker mode
tts.tts(
    text="...",
    speaker="Aaron Dreschner",  # Use speaker parameter
    language="en"
)

# Voice cloning mode
tts.tts(
    text="...",
    speaker_wav="/tmp/reference.wav",  # Use speaker_wav parameter
    language="en"
)
```

**Never mix**: `speaker` and `speaker_wav` are mutually exclusive

### 4.2 Chunking Strategy

**XTTS Warning**: Text >250 chars produces quality degradation warning

**Auto-chunking** (inherit from story_reels):
```python
from tts_v2.pipeline.chunker import chunk_text
from tts_v2.interfaces import ChunkingSpec

spec = ChunkingSpec(
    max_chars=200,  # Safe limit (XTTS threshold is 250)
    max_words=60,
    preserve_sentence_boundaries=True
)
chunks = chunk_text(long_text, spec)
```

**Stitching** (use existing story_reels logic):
- Apply 40ms crossfade between chunks
- Normalize loudness across stitched audio
- Return single WAV file

### 4.3 Error Handling

**Common Failure Modes**:
1. **Invalid speaker_id**: Return 400 + list of valid speakers
2. **Unsupported language**: Return 400 + list of supported languages
3. **Model load failure**: Retry once, then 500
4. **OOM during inference**: Log error, return 500 (increase container RAM)
5. **Reference audio quality issues**: Return 400 with actionable feedback

**Logging**: Use `structlog` (matches story_reels pattern)
```python
logger.info("tts.inference.start", speaker=speaker_id, text_length=len(text))
logger.error("tts.inference.failed", error=str(exc), speaker=speaker_id)
```

---

## 5. API Response Formats

### 5.1 Success Responses

**Audio Responses** (TTS, Voice Clone):
- Content-Type: `audio/wav`
- Headers:
  - `X-Sample-Rate: 22050`
  - `X-Duration-Sec: 3.45`
  - `X-Engine: coqui_xtts`
  - `X-Speaker: Aaron Dreschner` (TTS only)

**JSON Responses** (Speaker List):
- Content-Type: `application/json`
- Always include `count` and `last_updated`

### 5.2 Error Responses

**Standard Format**:
```json
{
  "error": {
    "code": "invalid_speaker",
    "message": "Speaker 'John Doe' not found",
    "valid_speakers": ["Aaron Dreschner", "..."],
    "request_id": "req_abc123"
  }
}
```

**HTTP Status Codes**:
- `400`: Client error (invalid input)
- `413`: Payload too large
- `429`: Rate limit exceeded (future)
- `500`: Server error (model failure)
- `503`: Service unavailable (model loading)

---

## 6. Dependencies & Environment

### 6.1 Python Requirements

```txt
TTS==0.22.0  # Coqui XTTS v2
torch==2.1.0
torchaudio==2.1.0
numpy==1.24.0
structlog==23.1.0
fastapi==0.108.0  # For Modal web endpoints
pydantic==2.5.0
```

### 6.2 Modal Configuration

**Image Setup**:
```python
image = (
    modal.Image.debian_slim()
    .pip_install(
        "TTS==0.22.0",
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "structlog==23.1.0",
        "fastapi[standard]==0.108.0",  # Includes uvicorn
        "pydantic==2.5.0",
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg")  # Required by Coqui
)
```

**GPU Requirements**:
- GPU: T4 (16GB VRAM) - sufficient for XTTS v2
- Alternative: A10G (24GB) if batch inference needed
- CPU: 4 cores, 16GB RAM for model loading

**Volume Setup** (Volume v2):
```python
volume = modal.Volume.from_name("coqui-models-v2", create_if_missing=True)
# Volume v2 features:
# - Distributed across all Modal regions
# - Higher throughput, improved random-access
# - True high-concurrency writes (100+ containers)
# - Faster commits and reloads
```

**ASGI App Setup**:
```python
from modal import App

app = App("coqui-apis")

@app.function(
    image=image,
    gpu=modal.gpu.T4(),
    volumes={"/models": volume},
    keep_warm=1,
    timeout=300,
    enable_memory_snapshot=True,  # Optional: faster cold starts
)
@modal.concurrent(max_inputs=10)  # Handle 10 concurrent requests
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    web_app = FastAPI(title="Coqui TTS API")
    # ... define routes ...
    return web_app
```

---

## 7. Migration from story_reels

### 7.1 Reusable Components

**Can be imported directly**:
- `tts_v2/engines/coqui_xtts.py` → Core engine logic
- `tts_v2/interfaces.py` → VoiceSpec, SynthesisRequest, etc.
- `tts_v2/pipeline/chunker.py` → Text chunking
- `tts_v2/pipeline/stitching.py` → Audio stitching
- `tts_v2/pipeline/postprocess.py` → Normalization, WAV wrapping

**Adapt for Modal**:
- Remove `TTSOrchestrator` (over-engineered for API)
- Simplify to direct engine calls
- Replace filesystem cache with Volume operations

### 7.2 Not Needed

- `cache/cache.py` → Use Modal Volume directly
- `cache/speaker_metadata.py` → Simplify to JSON read/write
- `engines/tortoise.py` → Out of scope for v1
- UI components → N/A for API

---

## 8. Testing Strategy

### 8.1 Unit Tests

1. **Speaker List Logic**:
   - Cache hit (<10 days)
   - Cache miss + rebuild
   - Force refresh
   - File corruption handling

2. **TTS Function**:
   - Valid speaker + valid language → audio
   - Invalid speaker → 400 error
   - Unsupported language → 400 error
   - Long text → chunked output

3. **Voice Clone Function**:
   - Valid reference audio → cloned audio
   - Missing reference → 400 error
   - Invalid audio format → 400 error

### 8.2 Integration Tests

1. **Modal Volume**:
   - Model loads from volume successfully
   - Speaker metadata persists across container restarts
   - Volume commit works correctly

2. **End-to-End**:
   - Deploy to Modal staging
   - Call all three APIs
   - Verify audio output quality
   - Check response times (<5s for TTS)

### 8.3 Load Tests

- Simulate 100 concurrent TTS requests
- Verify autoscaling works (should spawn 10 containers @ max_concurrent_inputs=10)
- Monitor costs per 1000 requests

---

## 9. Cost Estimation

**Assumptions**:
- TTS request: 3s inference time
- GPU: T4 @ $0.00051/s = $0.00153 per request
- Speaker List: CPU @ $0.0001/s, 0.1s = $0.00001 per request
- Volume storage: $0.10/GB/month = $0.18/month for 1.8GB model

**Monthly costs** (10k requests):
- TTS: 10k × $0.00153 = $15.30
- Speaker List: 10k × $0.00001 = $0.10
- Voice Clone: 1k × $0.00153 = $1.53
- Storage: $0.18
- **Total**: ~$17.11/month (plus ingress/egress)

**Optimization opportunities**:
- Audio caching: Reduce duplicate TTS calls by 50-70%
- Batch inference: Process multiple texts in parallel on GPU
- Spot instances: 70% cost reduction (if latency tolerance >30s)

---

## 10. Deployment Checklist

**Pre-deployment**:
- [ ] Model downloaded to Modal Volume
- [ ] Speaker metadata generated
- [ ] All tests passing
- [ ] FastAPI docs accessible (`/docs`)

**Post-deployment**:
- [ ] Health check endpoint returns 200
- [ ] All three APIs respond correctly
- [ ] Logs streaming to Modal dashboard
- [ ] Cost monitoring configured
- [ ] Rate limiting configured (future)

**Monitoring**:
- Track: Request count, error rate, p95 latency, GPU utilization
- Alerts: Error rate >5%, p95 latency >10s
- Dashboard: Grafana + Modal built-in metrics

---

## 11. Future Enhancements (Out of Scope for v1)

1. **Batch TTS**: Accept array of texts, return multiple audio files
2. **Audio Caching**: Hash(text + speaker + language) → cached WAV
3. **Streaming TTS**: Return audio chunks as they're generated
4. **Custom Voice Training**: Upload 10+ audio samples, fine-tune XTTS
5. **Emotion Control**: Add parameters for pitch, energy, emotion style
6. **Multi-speaker Dialogue**: Assign different speakers to dialogue turns
7. **SSML Support**: Parse SSML tags for prosody control
8. **STT Integration**: OpenAI Whisper APIs (separate planning doc)

---

## 12. Key Technical Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Modal Volume v2 (not NFS) | Distributed multi-region, 85% cold start reduction (60s → 8s) |
| Single ASGI app (not separate functions) | Shared GPU pool, model loaded once, cost-efficient |
| Stale-while-revalidate (10d TTL) | Balance freshness + compute costs (speakers are static) |
| Use `tts.speakers` attribute | No extra loading, instant access, proven in story_reels |
| Chunking at 200 chars | Avoid XTTS quality warning (250 threshold) |
| @modal.asgi_app() with FastAPI | Full routing control, auto OpenAPI docs, async handling |
| Reuse story_reels components | Proven chunking + stitching logic, production-tested |
| T4 GPU with keep_warm=1 | Sufficient for XTTS, <500ms response, 60% cheaper than A10G |
| Memory snapshotting (optional) | 60% faster cold starts (8s → 3s), minimal storage cost |
| @modal.concurrent(max_inputs=10) | Async request handling, efficient GPU utilization |

---

## 13. References

**Modal Documentation**:
- Volumes: https://modal.com/docs/guide/volumes
- GPU functions: https://modal.com/docs/guide/gpu
- Web endpoints: https://modal.com/docs/guide/webhooks

**Coqui Documentation**:
- XTTS v2: https://github.com/coqui-ai/TTS
- SpeakerManager: TTS/tts/utils/speakers.py
- Model files: ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/

**story_reels Reference**:
- Engine: `/Users/abhiroopprasad/code/clients/story_reels/tts_v2/engines/coqui_xtts.py`
- Interfaces: `/Users/abhiroopprasad/code/clients/story_reels/tts_v2/interfaces.py`
- UI example: `/Users/abhiroopprasad/code/clients/story_reels/ui/pages/1_Voice_Sandbox.py`

---

**Next Steps**: Create execution document with granular task breakdown, status tracking, and implementation sequence.
