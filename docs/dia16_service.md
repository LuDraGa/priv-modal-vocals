# Dia 1.6B Service

The Dia16 service adds Nari Labs `Dia-1.6B-0626` as an independent Modal-hosted
high-fidelity batch TTS and dialogue API. It is separate from Dia2 because the
model runtime, conditioning method, sample rate, and dependencies are different.

## Hosting Model

- Modal app: `dia16-tts-api`
- Modal Volume: `dia16-models-v1`
- Checkpoint: `nari-labs/Dia-1.6B-0626`
- Predefined voices: 43 third-party audio prompt profiles from
  `devnen/Dia-TTS-Server`
- GPU default: L40S
- Runtime mode: batch WAV
- Output sample rate: 44.1 kHz
- Scale behavior: `min_containers=0`, short scaledown window

Dia16 is English-only in this v1. The current official model card lists English
generation only, so profile language is restricted to `en` until Nari publishes
multilingual support for this checkpoint. Realtime streaming and voice
conversion are out of scope.

## Setup

Download model assets once:

```bash
modal run dia16_service/download_models.py
```

This materializes the model into:

- `/models/dia16/local/Dia-1.6B-0626`
- `/models/dia16/local/dac_44khz`
- `/models/dia16/predefined_voices`

Dia16's processor depends on Descript Audio Codec. The downloader preloads
`descript/dac_44khz` and rewrites `audio_tokenizer_config.json` to point to the
local DAC path so runtime can stay offline.

Serve a dev endpoint:

```bash
modal serve dia16_service/main.py
```

Deploy:

```bash
modal deploy dia16_service/main.py
```

## API

### `GET /health`

Checks service status and model load state.

### `GET /api-info`

Returns endpoint usage, inputs, output headers, limits, and Dia16-specific
conditioning notes.

### `POST /tts`

Simple single-speaker TTS.

```json
{
  "text": "Hello from Dia sixteen.",
  "voice_profile_id": "optional-profile-id",
  "predefined_voice_id": "optional-predefined-id",
  "style": "optional short style hint",
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "cfg_scale": 2.0,
  "max_new_tokens": 1024,
  "seed": 1234
}
```

`cfg_scale` is accepted for API compatibility with Dia2. The Transformers Dia16
generation path may ignore it.

Use either `voice_profile_id` or `predefined_voice_id`, not both. Predefined
voice ids come from `GET /predefined-voice-profiles`.

### `POST /dialogue`

Dia-native `[S1]` / `[S2]` dialogue generation.

```json
{
  "script": "[S1] Hi there.\n[S2] Hello!",
  "speaker_profiles": {
    "S1": "profile-id-1"
  },
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "max_new_tokens": 1024,
  "seed": 1234
}
```

When `speaker_profiles` is provided, v1 uses the first/S1 profile as the audio
prompt for the generation.

### `POST /tts-with-upload`

One-shot voice-conditioned TTS without saving a profile.

Required multipart fields:

- `text`
- `reference_audio`: WAV, 5-20 seconds. 5-10 seconds is recommended for best
  cloning quality.
- `reference_transcript`: exact transcript of the reference audio. It must start
  with `[S1]`.

Dia16 conditioning differs from Dia2: the engine passes `reference_audio` as an
audio prompt and prepends `reference_transcript` to the generated script. The
decoded output excludes the audio prompt continuation.

### Voice Profile APIs

Profiles are Dia16-specific caller-created reference voices, not built-in voices
and not shared with Dia2.

- `POST /voice-profiles`: create a reusable profile.
- `GET /voice-profiles`: list profiles for dropdowns and filters.
- `GET /voice-profiles/{id}`: inspect a profile.
- `DELETE /voice-profiles/{id}`: delete profile metadata and reference audio.

`consent_confirmed=true` is required before storing voice data.

### `GET /predefined-voice-profiles`

Lists the preloaded predefined voice prompt profiles. These are not native Nari
model voices; Nari's Dia 1.6B model card says the model was not fine-tuned on a
specific voice. They are third-party curated WAV/TXT audio prompts that make
voice selection easier without manual upload.

Reference profile quality rules:

- Use a 5-20 second reference clip.
- Prefer 5-10 seconds for cloning quality.
- Store the exact reference transcript with the `[S1]` speaker tag at the start.
- The profile registry stores this transcript alongside the WAV, serving the
  same purpose as a transcript sidecar file.

## Test Notes

Required checks:

- `ruff check dia16_service`
- `PYTHONPYCACHEPREFIX=/private/tmp/dia_pycache python3 -m compileall dia16_service`
- `modal run dia16_service/download_models.py`
- `modal deploy dia16_service/main.py`
- `GET /health`
- `GET /api-info`
- `POST /tts`
- `POST /dialogue`
- `POST /tts-with-upload`
- `POST /voice-profiles`, then `/tts` with `voice_profile_id`
- `GET /predefined-voice-profiles`, then `/tts` with `predefined_voice_id`
- `DELETE /voice-profiles/{id}` after test profiles are no longer needed

Acceptance:

- Runtime logs do not show Hugging Face download attempts.
- Audio responses are valid WAV files at 44.1 kHz.
- Response headers include `X-Engine: dia16`, `X-Model: Dia-1.6B-0626`,
  `X-Compute-Sec`, and `X-Duration-Sec`.
