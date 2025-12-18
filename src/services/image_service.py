import os
from openai import OpenAI
from typing import Dict, Any, Optional
import base64
from datetime import datetime
from pathlib import Path

class ImageService:
    def __init__(self):
        self.client = None
        self.image_dir = Path("images")
        self.image_dir.mkdir(exist_ok=True)

    def _get_client(self):
        if not self.client:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        return self.client

    def generate_image(self, prompt: str, model: str = "gpt-image-1") -> Dict[str, Any]:
        """Generate an image using OpenAI."""
        client = self._get_client()
        if not client:
             return {"success": False, "error": "OpenAI API key not found"}

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            result = client.images.generate(
                model=model,
                prompt=prompt[:1000],
                n=1,
                response_format="b64_json"
            )

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            image_path = self.image_dir / f"generated_{timestamp}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            return {
                "success": True,
                "image_path": str(image_path),
                "timestamp": timestamp
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
