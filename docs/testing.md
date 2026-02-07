# Testing Guide

This guide covers pre-deployment testing procedures for all services in the Modal APIs project.

## Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure Modal CLI is authenticated
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
```

---

## Coqui TTS Service

### 1. Local Development Server

Start the development server:

```bash
modal serve coqui_service/main.py
```

Expected output:
- Dev endpoint: `https://[username]--coqui-apis-fastapi-app-dev.modal.run`
- Container loads model from Volume (~15-20 seconds)
- Logs show: `tts_engine.initialized` and `fastapi.startup`

### 2. Health Check

```bash
curl -s https://[DEV_ENDPOINT]/health | jq
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "speakers_available": 58,
  "version": "0.1.0"
}
```

### 3. List Available Speakers

```bash
curl -s 'https://[DEV_ENDPOINT]/speakers' | jq '.speakers[:10]'
```

Expected: List of 58 speaker names (e.g., "Claribel Dervla", "Daisy Studious", etc.)

### 4. Text-to-Speech Test

```bash
curl -X POST https://[DEV_ENDPOINT]/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Coqui TTS API.",
    "speaker_id": "Claribel Dervla",
    "language": "en"
  }' \
  --output test_tts.wav
```

Verify audio:
```bash
# Check file properties
file test_tts.wav
ffmpeg -i test_tts.wav 2>&1 | grep -E "(Duration|Audio:)"

# Play audio (macOS)
afplay test_tts.wav

# Play audio (Linux)
aplay test_tts.wav
```

Expected:
- HTTP 200 status
- WAV file ~4-5 seconds duration
- 24kHz, 16-bit PCM, mono
- Clear speech output

### 5. Voice Cloning Test

#### Single Reference Audio

Prepare a reference audio file (WAV, MP3, or M4A format, <10MB, 3-30 seconds duration):

```bash
curl -X POST https://[DEV_ENDPOINT]/voice-clone \
  -F "text=This is a test of voice cloning technology." \
  -F "language=en" \
  -F "reference_audio=@path/to/reference_audio.wav" \
  --output test_clone_single.wav
```

#### Multiple Reference Audio (Better Quality)

For improved cloning quality, provide 2-5 reference audio files of the same speaker:

```bash
curl -X POST https://[DEV_ENDPOINT]/voice-clone \
  -F "text=This is a test of voice cloning with multiple references." \
  -F "language=en" \
  -F "reference_audio=@path/to/reference1.wav" \
  -F "reference_audio=@path/to/reference2.wav" \
  -F "reference_audio=@path/to/reference3.wav" \
  --output test_clone_multi.wav
```

Verify:
```bash
file test_clone_single.wav
afplay test_clone_single.wav  # macOS

# Check response headers for validation warnings
curl -X POST https://[DEV_ENDPOINT]/voice-clone \
  -F "text=Test" \
  -F "language=en" \
  -F "reference_audio=@reference.wav" \
  -I --output /dev/null
# Look for X-Validation-Warnings header
```

Expected:
- HTTP 200 status
- Voice matches reference audio characteristics
- Clear, natural-sounding speech
- Response headers include:
  - `X-Reference-Count`: Number of reference files used
  - `X-Validation-Warnings`: Any quality warnings (if applicable)
  - `X-Duration-Sec`: Generated audio duration

#### Reference Audio Requirements

**Duration:**
- Minimum: 3 seconds
- Optimal: 6-10 seconds (best quality)
- Maximum: 30 seconds

**Quality:**
- Sample rate: 16kHz minimum, 22kHz+ optimal
- Format: WAV, MP3, or M4A
- Channels: Mono preferred (stereo acceptable)
- File size: Max 10MB per file
- Content: Clean audio, single speaker, no background noise

**Validation Errors:**
- 400: Reference audio too short (<3s) or too long (>30s)
- 400: Sample rate too low (<16kHz)
- 413: File too large (>10MB)
- 400: Invalid audio format

### 6. API Documentation

Check interactive API docs:
```bash
open https://[DEV_ENDPOINT]/docs
```

Verify:
- OpenAPI/Swagger UI loads
- All endpoints documented (`/health`, `/speakers`, `/tts`, `/voice-clone`)
- Request/response schemas visible

---

## Whisper STT Service (Coming Soon)

### 1. Local Development

```bash
modal serve whisper_service/main.py
```

### 2. Health Check

```bash
curl -s https://[DEV_ENDPOINT]/health | jq
```

### 3. Speech-to-Text Test

```bash
curl -X POST https://[DEV_ENDPOINT]/transcribe \
  -F "audio=@test_audio.wav" \
  --output transcription.json
```

---

## Deployment Checklist

Before pushing to `main` branch or triggering deployment:

- [ ] All local tests pass
- [ ] Health check returns `healthy` status
- [ ] TTS generates clear audio
- [ ] Voice cloning works with sample reference
- [ ] No errors in Modal logs
- [ ] API documentation is accessible

---

## Troubleshooting

### Container Timeout on First Request

**Symptom**: First request times out after ~60s

**Cause**: Container is still loading the model from Volume

**Solution**: Wait 15-20 seconds after container starts before making requests

### Invalid Speaker Error

**Symptom**: 400 error "Invalid speaker"

**Cause**: Speaker name doesn't match available speakers

**Solution**:
```bash
# List available speakers
curl https://[DEV_ENDPOINT]/speakers | jq '.speakers'
```

### Audio Quality Issues

**Symptom**: Garbled or low-quality audio

**Cause**: Text too long (>200 chars per chunk)

**Solution**: API automatically chunks text, but verify input is reasonable length

### Model Not Found

**Symptom**: Error loading model from `/models/coqui`

**Cause**: Model not downloaded to Volume

**Solution**:
```bash
modal run coqui_service/download_models.py
```

---

## Performance Benchmarks

Expected performance (T4 GPU):

| Metric | Value |
|--------|-------|
| Cold start (no snapshot) | ~15-20s |
| Cold start (with snapshot) | ~3s |
| Synthesis time (50 chars) | ~3-4s |
| Sample rate | 24kHz |
| Audio format | 16-bit PCM WAV |

---

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [XTTS v2 Model Info](https://huggingface.co/coqui/XTTS-v2)
