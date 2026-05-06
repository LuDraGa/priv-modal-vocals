# Dia2 Service

The Dia2 service adds Nari Labs Dia2-1B as an independent Modal-hosted batch TTS
and dialogue API. It is intentionally separate from Coqui, WhisperX, and
OpenVoice so it can be tested, scaled, and deployed independently.

## Research Checklist

Before changing the Dia2 service, verify current upstream behavior:

- Dia2 repository and model cards: inference API, prompt format, prefix-audio
  conditioning, model limits, sample rate, license, and safety constraints.
- Modal hosting: GPU fit, cold start, model load time from Volume, generation
  time, storage footprint, and free-tier burn rate.
- Repo fit: preserve the `main.py`, `engine.py`, `routes.py`, `models.py`,
  `download_models.py` service pattern and path-filtered deploy workflow.
- Voice profiles: Dia2 has no fixed speaker catalog, so saved profiles are
  user/developer-created reference voices plus metadata.

## Hosting Model

- Modal app: `dia2-tts-api`
- Modal Volume: `dia2-models-v1`
- Checkpoint: `nari-labs/Dia2-1B`
- GPU default: T4
- Runtime mode: batch WAV
- Scale behavior: `min_containers=0`, short scaledown window

Dia2 is English-only in this v1. Realtime streaming is deferred because the
official server support is still maturing, and true voice conversion remains in
`vc_service`.

## Setup

Download model assets once:

```bash
modal run dia_service/download_models.py
```

Serve a dev endpoint:

```bash
modal serve dia_service/main.py
```

Deploy when explicitly needed:

```bash
modal deploy dia_service/main.py
```

## API

### `GET /health`

Checks service status and model load state.

### `GET /api-info`

Returns endpoint usage, inputs, output headers, limits, and implementation notes.

### `POST /tts`

Simple single-speaker TTS.

```json
{
  "text": "Hello from Dia2.",
  "voice_profile_id": "optional-profile-id",
  "style": "optional short style hint",
  "temperature": 0.8,
  "top_k": 50,
  "cfg_scale": 2.0,
  "seed": 1234
}
```

Returns `audio/wav` with `X-Compute-Sec`, `X-Duration-Sec`, `X-Model`, and
`X-Mode` headers.

### `POST /dialogue`

Dia2-native `[S1]` / `[S2]` dialogue generation.

```json
{
  "script": "[S1] Hi there.\n[S2] Hello!",
  "speaker_profiles": {
    "S1": "profile-id-1",
    "S2": "profile-id-2"
  },
  "temperature": 0.8,
  "top_k": 50,
  "cfg_scale": 2.0,
  "seed": 1234
}
```

### `POST /tts-with-upload`

One-shot voice-conditioned TTS without saving a profile.

Required multipart fields:

- `text`
- `reference_audio`
- `reference_transcript`

`reference_transcript` is stored/logged at the API boundary for audit and future
profile compatibility; the current Dia2 runtime transcribes prefix audio
internally.

### Voice Profile APIs

Profiles are caller-created reference voices, not built-in Dia2 voices.

- `POST /voice-profiles`: create a reusable profile.
- `GET /voice-profiles`: list profiles for dropdowns and filters.
- `GET /voice-profiles/{id}`: inspect a profile.
- `DELETE /voice-profiles/{id}`: delete profile metadata and reference audio.

Profile metadata:

- `name`
- `reference_audio`
- `reference_transcript`
- `gender`
- `accent`
- `language`
- `style_tags`
- `use_case`
- `quality_rating`
- `notes`
- `consent_confirmed`

`consent_confirmed=true` is required before storing voice data.

## Cost Notes

Modal free-tier planning baseline is roughly `$30/month`. GPU-only ceilings are
approximately T4 50.8 hours/month, L4 37.5 hours/month, and A10 27.2
hours/month before CPU, RAM, and storage.

Benchmark before production confidence:

- cold start time
- model load time from Volume
- generation time for 10s, 30s, and 60s outputs
- GPU memory usage
- cost per request from `X-Compute-Sec`

Keep the service scale-to-zero unless there is a deliberate paid decision to
trade cost for lower latency.
