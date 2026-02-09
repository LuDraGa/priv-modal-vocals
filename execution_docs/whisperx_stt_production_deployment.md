# WhisperX STT Service - Production Deployment

**Date**: 2026-02-09
**Status**: ✅ READY FOR PRODUCTION DEPLOYMENT

---

## Overview

This document tracks the production deployment setup for the WhisperX Speech-to-Text service.

## Completed Tasks

### 1. ✅ API Documentation Endpoint
**Status**: Implemented
**Date**: 2026-02-09

Added `/api-info` endpoint to provide comprehensive API documentation:
- **Endpoint**: `GET /api-info`
- **Models Added**: `APIEndpointInfo`, `APIInfoResponse` in `whisper_service/models.py`
- **Route Added**: In `whisper_service/routes.py` (lines 227-294)

Documented endpoints:
- `/transcribe` - POST with audio file upload
- `/languages` - GET list of 99 supported languages
- `/health` - GET service status and GPU availability
- `/api-info` - GET this documentation

### 2. ✅ GitHub Actions Workflow
**Status**: Configured
**Date**: 2026-02-09

Updated `.github/workflows/deploy.yml` with WhisperX deployment:
- **Added Path Filter**: `whisper_service/**` triggers workflow
- **Created Job**: `deploy-whisper` mirrors Coqui deployment pattern
- **Conditional Execution**: Only runs when `whisper_service/` changes
- **Manual Trigger**: Supports `workflow_dispatch` for manual deployments

Workflow features:
- Python 3.12 setup
- Modal CLI installation
- Automated deployment via `modal deploy whisper_service/main.py`
- Health check placeholder (10s stabilization period)

### 3. ✅ Dependency Compatibility Verification
**Status**: Verified
**Date**: 2026-02-09

**Coqui Service**:
- torch==2.4.1, torchaudio==2.4.1
- transformers<5.0
- coqui-tts[codec]>=0.27.3

**WhisperX Service**:
- torch 2.8.0 (from WhisperX GitHub install)
- WhisperX v3.7.6 from GitHub
- structlog>=25.5.0, fastapi[standard]>=0.108.0

**Result**: ✅ No conflicts - services run in isolated Modal containers with independent images.

---

## Deployment Architecture

### Modal Configuration
```python
@app.function(
    image=image,
    gpu="A10G",                  # 24GB VRAM (Whisper + Wav2Vec2)
    volumes={"/models": volume},
    min_containers=0,            # Scale to zero
    timeout=600,                 # 10 min (long audio)
    enable_memory_snapshot=True, # Fast cold starts (~5-8s)
    scaledown_window=120,        # Keep alive 2min
)
```

### Resources
- **GPU**: A10G (24GB VRAM)
- **Volume**: `whisperx-models-v1` (~4GB models)
- **Model**: WhisperX large-v3-turbo
- **Alignment**: Wav2Vec2 (English pre-loaded, others on-demand)

---

## API Endpoints

### Production Endpoints

1. **POST /transcribe**
   - Upload audio file (WAV, MP3, M4A, FLAC)
   - Optional language parameter
   - Returns word-level timestamps

2. **GET /languages**
   - Returns 99 supported languages

3. **GET /health**
   - Service status, GPU availability

4. **GET /api-info**
   - Comprehensive API documentation

---

## Next Steps (Production Deployment)

### Pre-Deployment Checklist
- [ ] Verify Modal Volume `whisperx-models-v1` has models downloaded
- [ ] Confirm Modal secrets configured in GitHub Actions
- [ ] Test `/api-info` endpoint in local dev server
- [ ] Review deployment logs for any warnings

### Deployment Steps

1. **Test Local Dev Server**
   ```bash
   source .venv/bin/activate
   python3 -m modal serve whisper_service/main.py
   ```
   - Verify `/api-info` endpoint works
   - Test transcription with sample audio

2. **Merge to Main Branch**
   ```bash
   git add .
   git commit -m "Add API documentation and production deployment workflow for WhisperX STT"
   git push origin main
   ```

3. **Monitor GitHub Actions**
   - Watch workflow run: https://github.com/[user]/modal_apis/actions
   - Verify `deploy-whisper` job succeeds
   - Check deployment logs

4. **Post-Deployment Verification**
   ```bash
   # Get production URL from Modal dashboard
   curl "https://[PRODUCTION-ENDPOINT]/health" | jq
   curl "https://[PRODUCTION-ENDPOINT]/api-info" | jq
   ```

---

## Differences from Coqui Service

| Aspect | Coqui TTS | WhisperX STT |
|--------|-----------|--------------|
| GPU | T4 (16GB) | A10G (24GB) |
| Timeout | 300s (5min) | 600s (10min) |
| Volume | `coqui-models-v2` | `whisperx-models-v1` |
| Model Size | ~1.8GB | ~4GB |
| Cold Start | ~3s | ~5-8s |

---

## Known Issues & Workarounds

### PyTorch 2.8.0 weights_only Security
**Issue**: PyTorch 2.6+ has strict `weights_only=True` by default, blocking pyannote models

**Solution**: Monkey-patched `torch.load` to force `weights_only=False` for trusted HuggingFace models
- Applied in: `whisper_service/engine.py` (lines 167-180)
- Applied in: `whisper_service/download_models.py` (lines 53-66)

---

## Success Metrics

### Performance Benchmarks (from Testing)
- **Audio Duration**: 3 minutes
- **Transcription Time**: 38 seconds
- **Accuracy**: Excellent (complex narrative transcribed correctly)
- **Word-level Timestamps**: Working (each word has start/end/score)

---

## Rollback Plan

If deployment fails:
1. GitHub Actions will show error in logs
2. Previous deployment remains active (Modal preserves last working version)
3. Fix issues and re-run workflow via manual trigger
4. Or revert git commit and push to main

---

## Documentation References

- [Modal Apps Documentation](https://modal.com/docs)
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [VERIFICATION_SUMMARY.md](../VERIFICATION_SUMMARY.md)
- [DEPENDENCIES_SUMMARY.md](../DEPENDENCIES_SUMMARY.md)

---

**Updated**: 2026-02-09
**Next Review**: After first production deployment
