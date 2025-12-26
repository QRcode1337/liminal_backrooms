import os
import requests
import json
import base64
import re
import logging
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_image_from_text(text, model="google/gemini-3-pro-image-preview"):
    """Generate an image based on text using OpenRouter's image generation API"""
    try:
        image_dir = Path("images")
        image_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ],
            "modalities": ["image", "text"],
            "max_tokens": 1024
        }

        print(f"Generating image with {model}...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("choices"):
                message = result["choices"][0].get("message", {})

                if message.get("images"):
                    for image in message["images"]:
                        image_url = image["image_url"]["url"]
                        print(f"Generated image URL (first 50 chars): {image_url[:50]}...")

                        if image_url.startswith('data:image'):
                            try:
                                ext = ".jpg"
                                if image_url.startswith('data:image/png'):
                                    ext = ".png"
                                elif image_url.startswith('data:image/gif'):
                                    ext = ".gif"
                                elif image_url.startswith('data:image/webp'):
                                    ext = ".webp"

                                base64_data = image_url.split(',', 1)[1] if ',' in image_url else image_url

                                image_data = base64.b64decode(base64_data)
                                image_path = image_dir / f"generated_{timestamp}{ext}"
                                with open(image_path, "wb") as f:
                                    f.write(image_data)

                                print(f"Generated image saved to {image_path}")
                                return {
                                    "success": True,
                                    "image_path": str(image_path),
                                    "timestamp": timestamp
                                }
                            except Exception as e:
                                print(f"Failed to decode base64 image: {e}")
                                return {
                                    "success": False,
                                    "error": f"Failed to decode image: {e}"
                                }
                        else:
                            try:
                                img_response = requests.get(image_url, timeout=30)
                                if img_response.status_code == 200:
                                    image_path = image_dir / f"generated_{timestamp}.png"
                                    with open(image_path, "wb") as f:
                                        f.write(img_response.content)

                                    print(f"Generated image saved to {image_path}")
                                    return {
                                        "success": True,
                                        "image_path": str(image_path),
                                        "timestamp": timestamp
                                    }
                            except Exception as e:
                                print(f"Failed to download image: {e}")
                                return {
                                    "success": False,
                                    "error": f"Failed to download image: {e}"
                                }

                print(f"No images in response. Message keys: {list(message.keys()) if isinstance(message, dict) else 'non-dict'}")
                return {
                    "success": False,
                    "error": "No images in API response"
                }
            else:
                return {
                    "success": False,
                    "error": "No choices in API response"
                }
        else:
            error_msg = f"API error {response.status_code}: {response.text[:500]}"
            print(f"Error generating image: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    except Exception as e:
        print(f"Error generating image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def ensure_videos_dir() -> Path:
    """Create a 'videos' directory in the project root if it doesn't exist."""
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    return videos_dir

def generate_video_with_sora(
    prompt: str,
    model: str = "sora-2",
    seconds: int | None = None,
    size: str | None = None,
    poll_interval_seconds: float = 5.0,
) -> dict:
    """
    Create a Sora video via REST API, poll until completion, and save MP4 to videos/.
    """
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {"success": False, "error": "OPENAI_API_KEY not set"}

        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        verbose = os.getenv('SORA_VERBOSE', '1').strip() == '1'
        def vlog(msg: str):
            if verbose:
                print(msg)
        headers_json = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Start render job
        payload = {"model": model, "prompt": prompt}
        if seconds is not None:
            payload["seconds"] = str(seconds)
        if size is not None:
            payload["size"] = size

        create_url = f"{base_url}/videos"
        vlog(f"[Sora] Create: url={create_url} model={model} seconds={seconds} size={size}")
        vlog(f"[Sora] Prompt (truncated): {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        resp = requests.post(create_url, headers=headers_json, json=payload, timeout=60)
        if not resp.ok:
            err_text = resp.text
            try:
                err_json = resp.json()
                vlog(f"[Sora] Create error JSON: {err_json}")
            except Exception:
                vlog(f"[Sora] Create error TEXT: {err_text}")
            return {"success": False, "error": f"Create failed {resp.status_code}: {err_text}"}
        job = resp.json()
        video_id = job.get('id')
        status = job.get('status')
        vlog(f"[Sora] Job started: id={video_id} status={status}")
        if not video_id:
            return {"success": False, "error": "No video id returned from create()"}

        # Poll until completion/failed
        retrieve_url = f"{base_url}/videos/{video_id}"
        last_status = status
        last_progress = None
        while status in ("queued", "in_progress"):
            time.sleep(poll_interval_seconds)
            r = requests.get(retrieve_url, headers=headers_json, timeout=60)
            if not r.ok:
                vlog(f"[Sora] Retrieve failed: code={r.status_code} body={r.text}")
                return {"success": False, "video_id": video_id, "error": f"Retrieve failed {r.status_code}: {r.text}"}
            job = r.json()
            status = job.get('status')
            progress = job.get('progress')
            if status != last_status or progress != last_progress:
                vlog(f"[Sora] Status update: status={status} progress={progress}")
                last_status = status
                last_progress = progress

        if status != "completed":
            vlog(f"[Sora] Final non-completed status: {status} job={job}")
            return {"success": False, "video_id": video_id, "status": status, "error": f"Final status: {status}"}

        # Download the MP4
        content_url = f"{base_url}/videos/{video_id}/content"
        vlog(f"[Sora] Download: url={content_url}")
        rc = requests.get(content_url, headers={'Authorization': f'Bearer {api_key}'}, stream=True, timeout=300)
        if not rc.ok:
            vlog(f"[Sora] Download failed: code={rc.status_code} body={rc.text}")
            return {"success": False, "video_id": video_id, "status": status, "error": f"Download failed {rc.status_code}: {rc.text}"}

        videos_dir = ensure_videos_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_snippet = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt[:40]) or "video"
        out_path = videos_dir / f"{timestamp}_{safe_snippet}.mp4"
        with open(out_path, "wb") as f:
            for chunk in rc.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        vlog(f"[Sora] Saved video: {out_path}")
        return {
            "success": True,
            "video_id": video_id,
            "status": status,
            "video_path": str(out_path)
        }
    except Exception as e:
        logging.exception("Sora video generation error")
        return {"success": False, "error": str(e)}

def call_claude_vision_api(image_url):
    """Have Claude analyze the generated image"""
    from anthropic import Anthropic
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    try:
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail. What works well and what could be improved?"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url
                        }
                    }
                ]
            }]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error in vision analysis: {e}")
        return None
