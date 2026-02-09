# Modal APIs - Dependencies Summary

**Date**: 2026-02-09

## Shared Dependencies (Both Services)

Both `coqui_service` and `whisper_service` run in the same project and must use compatible dependencies.

### Core Dependencies

| Package | Version | Reason |
|---------|---------|--------|
| **torch** | `==2.4.1` | Pin to 2.4.1 (last version before weights_only=True default) |
| **torchaudio** | `==2.4.1` | Must match torch version |
| **transformers** | `<5.0` | Coqui requires 4.x, WhisperX compatible |
| **numpy** | `<2.4` | Both services compatible |
| **structlog** | `>=25.5.0` | Logging |
| **fastapi[standard]** | `>=0.108.0` | API framework |
| **pydantic** | `>=2.5.0` | Data validation |
| **python-multipart** | latest | File uploads |

### PyTorch 2.5+ Issue

**Problem**: PyTorch 2.5+ has strict `weights_only=True` by default, which blocks pyannote.audio VAD models.

**Error**:
```
UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.listconfig.ListConfig
```

**Solution**: Pin to `torch==2.4.1` and `torchaudio==2.4.1`

---

## Service-Specific Dependencies

### Coqui TTS Service

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",  # Pinned for compatibility
        "torchaudio==2.4.1",
        "transformers<5.0",
        "coqui-tts[codec]>=0.27.3",
        "structlog>=25.5.0",
        "numpy<2.4",
        "fastapi[standard]>=0.108.0",
        "pydantic>=2.5.0",
        "python-multipart",
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg")
    .add_local_python_source("coqui_service")
)
```

### WhisperX STT Service

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",  # Pinned for compatibility
        "torchaudio==2.4.1",
        "transformers>=4.36.0",
        "faster-whisper>=0.10.0",
        "pyannote.audio==3.1.1",
        "structlog>=25.5.0",
        "numpy<2.4",
        "fastapi[standard]>=0.108.0",
        "pydantic>=2.5.0",
        "python-multipart",
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg git")
    .pip_install("git+https://github.com/m-bain/whisperX.git@v3.1.5")
    .add_local_python_source("whisper_service")
)
```

---

## Version Pinning Strategy

1. **torch/torchaudio**: `==2.4.1` (strict pin for pyannote compatibility)
2. **transformers**: `<5.0` (Coqui requires 4.x)
3. **pyannote.audio**: `==3.1.1` (tested with torch 2.4.1)
4. **WhisperX**: From GitHub `@v3.1.5` (official, not yanked PyPI version)

---

## Testing Checklist

- [ ] Verify torch version in both services: `torch==2.4.1`
- [ ] Test Coqui TTS model loading
- [ ] Test WhisperX + pyannote VAD loading
- [ ] Confirm no dependency conflicts
- [ ] Test both services can coexist in same project

---

## Future Considerations

When upgrading to PyTorch 2.5+:
- Requires pyannote.audio fix for `weights_only=True`
- Or explicitly set `weights_only=False` (security risk)
- Monitor: https://github.com/pyannote/pyannote-audio/issues
