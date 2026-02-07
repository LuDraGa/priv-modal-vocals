# Modal APIs - Claude Code Instructions

Project-specific instructions for working with the Modal APIs voice stack.

## Project Overview

This repository contains Modal-hosted voice processing APIs:
- **Coqui TTS Service**: Text-to-Speech and Voice Cloning (XTTS v2)
- **Whisper STT Service**: Speech-to-Text (planned)

## Testing Requirements

**IMPORTANT**: Always test locally before deploying to production.

See [Testing Guide](docs/testing.md) for comprehensive pre-deployment testing procedures including:
- Health checks
- TTS endpoint testing
- Voice cloning validation
- Performance benchmarks
- Troubleshooting guide

Quick test command:
```bash
modal serve coqui_service/main.py
curl https://[dev-endpoint]/health | jq
```

## Deployment

Deployments are automated via GitHub Actions:
- Push to `main` with changes to `coqui_service/` triggers TTS deployment
- Manual deployments: GitHub Actions → "Deploy Modal APIs" → Run workflow

## Project Structure

```
modal_apis/
├── coqui_service/          # TTS + Voice Cloning API
│   ├── main.py            # Modal app entry point
│   ├── engine.py          # TTS engine wrapper
│   ├── routes.py          # FastAPI endpoints
│   ├── models.py          # Pydantic models
│   └── utils/             # Audio processing, caching
├── whisper_service/        # STT API (future)
├── docs/                   # Documentation
│   ├── testing.md         # Testing procedures
│   └── execution_docs/    # Development logs
└── .github/workflows/      # CI/CD pipelines
```

## Key Technologies

- **Modal**: Serverless GPU compute platform
- **Coqui TTS**: XTTS v2 multilingual TTS model (58 speakers)
- **FastAPI**: Modern Python web framework
- **Python 3.12**: Project runtime
- **uv**: Package manager

## Development Workflow

1. Make changes locally
2. Test with `modal serve` (see [Testing Guide](docs/testing.md))
3. Commit and push to `main` branch
4. GitHub Actions auto-deploys to Modal

## Important Notes

- Model weights stored in Modal Volume `coqui-models-v2` (1.8GB)
- First-time setup requires running `modal run coqui_service/download_models.py`
- Memory snapshots enabled in production for fast cold starts (~3s)
- Independent service deployments prevent unnecessary redeployments
