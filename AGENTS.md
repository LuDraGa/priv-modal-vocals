# Modal APIs - Agent Instructions

These instructions apply to the whole repository. Use them when making code, test,
documentation, or deployment changes in this Modal-hosted voice API stack.

## Project Overview

This repository contains independent Modal services for voice processing:

- `coqui_service/`: Coqui XTTS v2 text-to-speech and voice cloning API
- `whisper_service/`: WhisperX speech-to-text API with word-level timestamps
- `vc_service/`: OpenVoice v2 voice conversion API
- `dia_service/`: Dia2-1B expressive TTS, dialogue, and reusable voice profiles
- `shared/`: Shared audio utilities used across services

Each service is designed to deploy independently so unrelated services do not
redeploy when one API changes.

## Agent Workflow

1. Inspect the service-specific files before changing behavior.
2. Keep changes scoped to the affected service and shared modules that are
   genuinely needed.
3. Preserve public API contracts unless the task explicitly asks to change them.
4. Update tests or docs when behavior, endpoints, request fields, response
   headers, model setup, or deployment steps change.
5. Test locally before deployment-oriented changes. See `docs/testing.md`.
6. Do not run production deploys unless explicitly asked.

## Common Commands

Install dependencies:

```bash
uv pip install -e ".[dev]"
```

Serve a Modal development endpoint:

```bash
modal serve coqui_service/main.py
modal serve whisper_service/main.py
modal serve vc_service/main.py
modal serve dia_service/main.py
```

Run one-time model downloads to Modal Volumes:

```bash
modal run coqui_service/download_models.py
modal run whisper_service/download_models.py
modal run vc_service/download_models.py
modal run dia_service/download_models.py
```

Run local quality checks when applicable:

```bash
ruff check .
pytest
```

## Testing Expectations

Always prefer testing the affected service through `modal serve` before changing
deployment workflows or production-facing behavior.

Minimum checks by service:

- Coqui TTS: `GET /health`, `GET /speakers`, `POST /tts`, and `POST /voice-clone`
- WhisperX STT: `GET /health`, `GET /languages`, and `POST /transcribe`
- Voice conversion: `GET /health` and `POST /voice-convert`
- Dia2 TTS: `GET /health`, `POST /tts`, `POST /dialogue`, and profile CRUD

For audio-generating endpoints, verify both the HTTP response and the generated
audio file properties. The testing guide includes example `curl`, `file`, and
`ffmpeg` commands.

## Deployment Notes

Deployments are handled by `.github/workflows/deploy.yml`.

- Pushes to `main` deploy services based on changed paths.
- `coqui_service/**` and `shared/**` can trigger Coqui deployment.
- `whisper_service/**` can trigger WhisperX deployment.
- `vc_service/**` and `shared/**` can trigger voice conversion deployment.
- `dia_service/**` and `shared/**` can trigger Dia2 deployment.
- Manual deployment is available through GitHub Actions workflow dispatch.

Direct deploy commands exist, but should only be run when requested:

```bash
modal deploy coqui_service/main.py
modal deploy whisper_service/main.py
modal deploy vc_service/main.py
modal deploy dia_service/main.py
```

## Important Implementation Details

- Runtime target is Python 3.12; Ruff is configured with a 100-character line
  length and selected lint rules in `pyproject.toml`.
- Coqui model weights live in Modal Volume `coqui-models-v2`.
- Voice conversion model weights live in Modal Volume `openvoice-models-v1`.
- Dia2 model weights and voice profiles live in Modal Volume `dia2-models-v1`.
- Production services use Modal memory snapshots for faster cold starts.
- Audio utility changes in `shared/audio.py` may affect more than one service.
  Check all consumers before editing shared behavior.
- Keep service package boundaries clear. Avoid importing implementation details
  across service directories unless they belong in `shared/`.

## Reference Docs

- `README.md`: feature overview, setup, and API examples
- `docs/testing.md`: pre-deployment testing procedures
- `execution_docs/`: implementation notes and service-specific history
