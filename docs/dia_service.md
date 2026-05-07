# Dia2 Service

The Dia2 service adds Nari Labs Dia2 as an independent Modal-hosted batch TTS
and dialogue API. It supports Dia2-1B by default and Dia2-2B as an experimental
opt-in model. It is intentionally separate from Coqui, WhisperX, and OpenVoice
so it can be tested, scaled, and deployed independently.

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
- Checkpoints: `nari-labs/Dia2-1B`, `nari-labs/Dia2-2B`
- GPU default: L40S
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

The downloader materializes model files into direct Volume paths:

- `/models/dia2/local/Dia2-1B`
- `/models/dia2/local/Dia2-2B`
- `/models/dia2/local/mimi`
- `/models/dia2/local/whisper-large-v3`

The Dia2-2B model reuses the same Mimi codec assets, so only the Dia2 checkpoint
adds extra storage and download time.

Prefix conditioning also needs `openai/whisper-large-v3`. Dia2 pulls this in
only when prefix audio is used: `Dia2.generate(...)` builds a prefix plan,
`dia2/runtime/voice_clone.py` lazily imports `whisper_timestamped`, then calls
`wts.load_model("openai/whisper-large-v3", ...)` to transcribe the prefix file.
Preloading it into the Volume keeps saved voice profiles compatible with
`TRANSFORMERS_OFFLINE=1`.

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
  "model_size": "1b",
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

`model_size` accepts:

- `1b`: default, lower cost and faster cold/warm requests.
- `2b`: experimental, heavier, likely better prosody/naturalness, higher cost.

### `POST /dialogue`

Dia2-native `[S1]` / `[S2]` dialogue generation.

```json
{
  "script": "[S1] Hi there.\n[S2] Hello!",
  "model_size": "1b",
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

Modal free-tier planning baseline is roughly `$30/month`. Approximate GPU-only
ceilings before CPU, RAM, and storage:

| GPU | Approx cost/hr | GPU hours from $30 |
| --- | ---: | ---: |
| A10 | $1.10 | 27.2 hr |
| L40S | $1.95 | 15.4 hr |
| A100 40GB | $2.10 | 14.3 hr |

L40S is the current Dia2 runtime default because it is a safer fit for Dia2-2B
than A10 while remaining close to A100 cost efficiency. Keep `min_containers=0`
unless there is a deliberate paid decision to keep the service warm.

Observed Dia2-1B on A10 before the L40S switch:

| Request | Total time | Compute time | Audio duration |
| --- | ---: | ---: | ---: |
| Cold short TTS | 39.95s | 16.19s | 4.64s |
| Warm short TTS | 31.07s | 10.99s | 3.84s |

Observed smoke tests after switching to L40S:

| Request | Total time | Compute time | Audio duration |
| --- | ---: | ---: | ---: |
| Dia2-1B short TTS | 21.43s | 7.04s | 2.96s |
| Dia2-2B tiny TTS | 12.45s | 5.40s | 1.92s |

The 2B number above proves the path loads and returns audio; it is not a full
benchmark because the prompt was shorter and the container was already warm.

Rough Dia2-2B planning estimates on L40S, assuming 2B lands around 1.5-2.5x the
1B runtime. These are planning numbers only; benchmark before treating them as
production costs.

| Request type | Est. L40S time | Est. cost/run | Runs from $30 |
| --- | ---: | ---: | ---: |
| Short warm TTS, ~4-6s audio | 30-55s | $0.016-$0.030 | ~1,000-1,845 |
| Short cold TTS | 60-100s | $0.033-$0.054 | ~555-920 |
| 15s audio | 80-140s | $0.043-$0.076 | ~395-690 |
| 30s audio | 150-260s | $0.081-$0.141 | ~213-369 |
| 60s audio | 300-520s | $0.163-$0.282 | ~106-184 |

Benchmark before production confidence:

- cold start time
- model load time from Volume
- generation time for 10s, 30s, and 60s outputs
- GPU memory usage
- cost per request from `X-Compute-Sec`

Benchmark both `model_size=1b` and `model_size=2b` after any GPU, dependency, or
Dia2 version change.
