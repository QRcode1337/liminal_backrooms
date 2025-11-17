# SORA Video Generation Guide

## ✅ Configuration Complete!

SORA environment variables have been added to your `.env` file.

## How to Use SORA

### Method 1: Manual Video Generation
1. In the GUI, select **"Sora 2"** or **"Sora 2 Pro"** as one of the AI models
2. Type a video prompt (e.g., "A serene lake at sunset with gentle waves")
3. Click "Propagate"
4. The system will:
   - Create a video generation job with OpenAI
   - Poll for completion (this takes 1-3 minutes)
   - Save the video to `videos/` folder
   - Display the result in the conversation

### Method 2: Auto-Generate from AI Responses
1. Set `SORA_AUTO_FROM_AI1=1` in `.env`
2. Select the **"Video Collaboration (AI-1 to Sora)"** conversation scenario
3. Use any text model for AI-1 (it will write cinematic descriptions)
4. AI-2 can be set to "Sora 2" or just left as another text model
5. AI-1's responses will automatically trigger video generation

## Environment Variables

```bash
# Enable/disable auto-generation from AI-1
SORA_AUTO_FROM_AI1=0          # 0=manual only, 1=auto-generate

# Model selection
SORA_MODEL=sora-2             # or sora-2-pro

# Video settings (optional)
SORA_SECONDS=                 # Leave empty for default (5-12 seconds)
SORA_SIZE=                    # Leave empty for default resolution

# Logging
SORA_VERBOSE=1                # 1=show detailed logs, 0=quiet
```

## Video Output

Generated videos are saved to:
```
/Users/patrickgallowaypro/Documents/PROJECTS/liminal_backrooms/videos/
```

Filename format: `YYYYMMDD_HHMMSS_prompt_snippet.mp4`

## Supported Video Durations
- Default: ~5 seconds
- Can specify: 5, 10, 12 seconds (set `SORA_SECONDS`)

## Supported Resolutions
- Default: OpenAI's default (typically 1280x720 or 1920x1080)
- Can specify custom size via `SORA_SIZE` (e.g., "1920x1080")

## Pricing Notes
⚠️ **SORA is a paid feature from OpenAI**
- Sora 2: ~$0.40-0.80 per 5 seconds
- Sora 2 Pro: Higher cost for better quality
- Check your OpenAI account for current pricing

## Troubleshooting

### "No module named 'openai'"
```bash
poetry install
```

### "OPENAI_API_KEY not set"
Make sure your `.env` file has a valid OpenAI API key.

### "Create failed 404"
Sora may not be available in your region or account. Check:
1. OpenAI account has Sora access
2. API key has proper permissions
3. Region restrictions

### Videos not appearing
Check the `videos/` directory:
```bash
ls -la videos/
```

### Long wait times
Video generation typically takes:
- 30-90 seconds for 5 seconds of video
- 60-180 seconds for 12 seconds of video

Watch the console for `[Sora]` log messages showing progress.

## Example Prompts

Good prompts are detailed and cinematic:
```
A close-up shot of a vintage typewriter, keys slowly pressing down
as invisible fingers type. Warm afternoon light streams through
a nearby window. Shallow depth of field, nostalgic mood.
```

```
Wide aerial shot of a misty forest at dawn. Camera slowly descends
through the canopy as birds take flight. Soft golden light filters
through the trees. Ethereal and peaceful atmosphere.
```

## Testing SORA

To test if SORA is working:
1. Restart the GUI: `poetry run python main.py`
2. Select "Sora 2" for AI-1
3. Type: "A red ball bouncing on a wooden floor"
4. Watch the console for `[Sora]` messages
5. Check the `videos/` folder after 1-2 minutes

## Need Help?

If SORA still doesn't work:
1. Check console output for error messages
2. Verify OpenAI API key is valid
3. Confirm Sora access in your OpenAI account
4. Check `videos/` folder permissions

Happy video generation! 🎬
