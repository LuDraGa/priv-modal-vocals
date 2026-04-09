# Voice Conversion Service — OpenVoice v2 Implementation

**Date:** 2026-04-09
**Scope:** New `vc_service/` Modal app + `shared/` audio utilities refactor

---

## Objective

Implement a standalone voice conversion API using OpenVoice v2 via the Coqui TTS library.
Voice conversion takes existing audio and converts the speaker's voice to match a reference voice,
preserving all content, emotion, rhythm, and style.

---

## Architecture Decisions

- **Separate Modal app** (`openvoice-vc-api`) — independent scaling and deployment from coqui_service
- **New Modal Volume** (`openvoice-models-v1`) — clean model isolation
- **GPU:** T4 (sufficient for VC inference)
- **Shared audio utilities** moved to `shared/audio.py` — consumed by both coqui_service and vc_service
- **Single endpoint** `/voice-convert` — focused, composable, source-agnostic

## I/O Contract

```
POST /voice-convert (multipart/form-data)
  source_audio      File      required    audio to convert
  reference_audio   File[]    1-3 files   target voice samples

Returns: audio/wav
  X-Sample-Rate: 24000
  X-Duration-Sec: float
  X-Engine: openvoice_v2
  X-Mode: voice_conversion
  X-Reference-Count: int
```

---

## Files Changed

### New Files
- [ ] `shared/audio.py` — all audio utils moved here from coqui_service, + validate_source_audio
- [ ] `vc_service/__init__.py`
- [ ] `vc_service/main.py` — Modal app definition
- [ ] `vc_service/engine.py` — VCEngine (load_model, convert)
- [ ] `vc_service/routes.py` — FastAPI /voice-convert + /health
- [ ] `vc_service/models.py` — Pydantic models
- [ ] `vc_service/utils/__init__.py`
- [ ] `vc_service/utils/audio.py` — passthrough re-export from shared
- [ ] `vc_service/download_models.py` — Modal download script

### Modified Files
- [ ] `coqui_service/utils/audio.py` — passthrough re-export from shared (backwards compat)
- [ ] `coqui_service/main.py` — add shared package to Modal image
- [ ] `pyproject.toml` — add vc_service to packages
- [ ] `.github/workflows/deploy.yml` — add vc_service conditional deployment

---

## Implementation Status

- [x] Execution doc created
- [x] shared/audio.py created
- [x] coqui_service/utils/audio.py updated to passthrough
- [x] coqui_service/main.py updated (shared package mount)
- [x] vc_service/ package structure created
- [x] VCEngine implemented
- [x] Routes implemented
- [x] Modal app configured
- [x] Download script created
- [x] pyproject.toml updated
- [x] deploy.yml updated

---

## Testing Plan (post-implementation)

1. `modal run vc_service/download_models.py` — download model to volume
2. `modal serve vc_service/main.py` — start local dev server
3. `curl [endpoint]/health` — confirm healthy
4. `curl -X POST [endpoint]/voice-convert -F "source_audio=@source.wav" -F "reference_audio=@ref.wav" --output converted.wav`
5. Listen to converted.wav — verify content preserved, voice changed

---

## Notes

- OpenVoice v2 uses a tone color encoder to extract a voice embedding from reference audio.
  Multiple reference files are concatenated for broader tone sampling.
- Sample rate: 24000Hz (hardcoded in OpenVoice v2 architecture)
- No streaming support — VC requires full source audio to process
- The vc_service is intentionally source-agnostic: source can be XTTS output, real recording, or any audio
