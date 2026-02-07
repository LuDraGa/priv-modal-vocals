# Setup Instructions - Run These Commands

Follow these steps in order to get your Coqui TTS API running on Modal.

## Step 1: Install Dependencies

Run this in the `modal_apis` directory:

```bash
# Install uv dependencies
uv pip install -e ".[dev]"

# Install Modal specifically
uv pip install modal
```

## Step 2: Authenticate with Modal

```bash
# Interactive Modal setup (opens browser for authentication)
python3 -m modal setup
```

After you complete the browser auth, run:

```bash
# Set your Modal token (you already have these credentials)
modal token set --token-id ak-9UTi3BBRvlnY8tLf22zF21 --token-secret as-vBlbcpc9MaRr2Fb33l8h2f
```

## Step 3: Download Models to Modal Volume

This downloads the Coqui XTTS v2 model (~1.8GB) to Modal Volume. **Run this once**:

```bash
modal run coqui_service/download_models.py
```

This will take 5-10 minutes. You'll see progress output.

## Step 4: Test Locally with Dev Endpoint

```bash
modal serve coqui_service/main.py
```

You'll see output like:
```
✓ Initialized. View run at https://modal.com/apps/...
✓ Created web function fastapi_app => https://yourname--coqui-apis-dev.modal.run
```

The dev endpoint URL (ends with `-dev.modal.run`) is now live and will auto-reload when you make code changes.

## Step 5: Test the Endpoints

Open the dev URL in your browser or test with curl:

### Health Check
```bash
curl https://yourname--coqui-apis-dev.modal.run/health
```

### List Speakers
```bash
curl https://yourname--coqui-apis-dev.modal.run/speakers
```

### TTS Synthesis
```bash
curl -X POST https://yourname--coqui-apis-dev.modal.run/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Coqui TTS API",
    "speaker_id": "Aaron Dreschner",
    "language": "en"
  }' \
  --output test.wav
```

Play the audio:
```bash
# macOS
afplay test.wav

# Linux
aplay test.wav
```

### Voice Clone
```bash
curl -X POST https://yourname--coqui-apis-dev.modal.run/voice-clone \
  -F "text=This is a test of voice cloning" \
  -F "language=en" \
  -F "reference_audio=@path/to/your/audio.wav" \
  --output cloned.wav
```

## Step 6: Deploy to Production

When you're ready to deploy the stable production endpoint:

```bash
modal deploy coqui_service/main.py
```

This creates a stable URL (without `-dev`): `https://yourname--coqui-apis.modal.run`

## Step 7: Set Up GitHub Auto-Deployment

1. Go to your GitHub repo: `https://github.com/YOUR_USERNAME/modal_apis`
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:
   - Name: `MODAL_TOKEN_ID`
   - Value: `ak-9UTi3BBRvlnY8tLf22zF21`
4. Click **New repository secret** again and add:
   - Name: `MODAL_TOKEN_SECRET`
   - Value: `as-vBlbcpc9MaRr2Fb33l8h2f`

Now, every push to `main` will automatically deploy your API!

## Development Workflow

### Making Changes

```bash
# 1. Create a feature branch
git checkout -b feature/my-change

# 2. Make your changes to the code

# 3. Test with dev endpoint (auto-reloads)
modal serve coqui_service/main.py

# 4. Commit and push
git add .
git commit -m "Add my change"
git push origin feature/my-change

# 5. Create PR on GitHub, review, merge to main

# 6. GitHub Actions automatically deploys to production!
```

### Checking Logs

```bash
# View Modal dashboard
open https://modal.com/apps

# Or view logs in terminal
modal app logs coqui-apis
```

## Troubleshooting

### "Model not found" error
Re-download the model:
```bash
modal run coqui_service/download_models.py
```

### "Invalid speaker" error
Check available speakers:
```bash
curl https://yourname--coqui-apis-dev.modal.run/speakers
```

### GitHub Actions failing
- Verify secrets are set correctly in GitHub repo settings
- Check the Actions tab for error details

## Next Steps

1. ✅ Test all three endpoints (`/health`, `/speakers`, `/tts`, `/voice-clone`)
2. ✅ Verify audio quality
3. ✅ Set up GitHub secrets for auto-deployment
4. ✅ Push to `main` and verify auto-deployment works
5. 🚀 Use your API in other projects!

## API Documentation

Once deployed, visit these URLs for interactive API docs:
- Swagger UI: `https://yourname--coqui-apis.modal.run/docs`
- ReDoc: `https://yourname--coqui-apis.modal.run/redoc`
