# shared_utils.py

import requests
import logging
import replicate
import openai
import time
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import base64
from together import Together
from openai import OpenAI
import re
from typing import Optional, Dict, List, Any, Union
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not found. Please install it with 'pip install beautifulsoup4'")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration constants
from config import (
    API_MAX_TOKENS, API_TEMPERATURE, API_TIMEOUT,
    LLAMA_MAX_TOKENS, LLAMA_TEMPERATURE, LLAMA_TOP_P, LLAMA_REPETITION_PENALTY, LLAMA_MAX_HISTORY,
    DEEPSEEK_MAX_TOKENS, DEEPSEEK_TEMPERATURE,
    TOGETHER_MAX_TOKENS, TOGETHER_TEMPERATURE, TOGETHER_TOP_P,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    SORA_POLL_INTERVAL_SECONDS
)

# Initialize Anthropic client with API key
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def call_claude_api(
    prompt: str,
    messages: List[Dict[str, Any]],
    model_id: str,
    system_prompt: Optional[str] = None
) -> str:
    """Call the Claude API with the given messages and prompt"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not found in environment variables"
    
    url = "https://api.anthropic.com/v1/messages"
    
    # Ensure we have a system prompt
    payload = {
        "model": model_id,
        "max_tokens": API_MAX_TOKENS,
        "temperature": API_TEMPERATURE
    }
    
    # Set system if provided
    if system_prompt:
        payload["system"] = system_prompt
        print(f"CLAUDE API USING SYSTEM PROMPT: {system_prompt}")
    
    # Clean messages to remove duplicates
    filtered_messages = []
    seen_contents = set()
    
    for msg in messages:
        # Skip system messages (handled separately)
        if msg.get("role") == "system":
            continue
            
        # Check for duplicates by content
        content = msg.get("content", "")
        if content in seen_contents:
            print(f"Skipping duplicate message in API call: {content[:30]}...")
            continue
            
        seen_contents.add(content)
        filtered_messages.append(msg)
    
    # Add the current prompt as the final user message
    filtered_messages.append({
        "role": "user",
        "content": prompt
    })

    # Add filtered messages to payload
    payload["messages"] = filtered_messages
    
    # Actual API call
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if 'content' in data and len(data['content']) > 0:
            for content_item in data['content']:
                if content_item.get('type') == 'text':
                    return content_item.get('text', '')
            # Fallback if no text type content is found
            logger.warning("No text type content found in Claude API response")
            return str(data['content'])
        logger.error("No content in Claude API response")
        return "No content in response"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Claude API: {e}")
        return f"Error calling Claude API: {str(e)}"
    except Exception as e:
        logger.exception(f"Unexpected error calling Claude API: {e}")
        return f"Error calling Claude API: {str(e)}"

def call_llama_api(
    prompt: str,
    conversation_history: List[Dict[str, Any]],
    model: str,
    system_prompt: str
) -> Optional[str]:
    # Only use the last N exchanges to prevent context length issues
    recent_history = conversation_history[-LLAMA_MAX_HISTORY:] if len(conversation_history) > LLAMA_MAX_HISTORY else conversation_history
    
    # Format the conversation history for LLaMA
    formatted_history = ""
    for message in recent_history:
        if message["role"] == "user":
            formatted_history += f"Human: {message['content']}\n"
        else:
            formatted_history += f"Assistant: {message['content']}\n"
    formatted_history += f"Human: {prompt}\nAssistant:"

    try:
        # Stream the output and collect it piece by piece
        response_chunks = []
        for chunk in replicate.run(
            model,
            input={
                "prompt": formatted_history,
                "system_prompt": system_prompt,
                "max_tokens": LLAMA_MAX_TOKENS,
                "temperature": LLAMA_TEMPERATURE,
                "top_p": LLAMA_TOP_P,
                "repetition_penalty": LLAMA_REPETITION_PENALTY
            },
            stream=True  # Enable streaming
        ):
            if chunk is not None:
                response_chunks.append(chunk)
                # Print each chunk as it arrives
                # print(chunk, end='', flush=True)
        
        # Join all chunks for the final response
        response = ''.join(response_chunks)
        return response
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error calling LLaMA: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error calling LLaMA API: {e}")
        return None

def call_openai_api(
    prompt: str,
    conversation_history: List[Dict[str, Any]],
    model: str,
    system_prompt: str
) -> Optional[str]:
    try:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=API_MAX_TOKENS,
            n=1,
            temperature=API_TEMPERATURE,
            stream=True
        )
        
        collected_messages = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:  # Changed condition
                collected_messages.append(chunk.choices[0].delta.content)
                
        full_reply = ''.join(collected_messages)
        return full_reply

    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error calling OpenAI API: {e}")
        return None

def call_openrouter_api(
    prompt: str,
    conversation_history: List[Dict[str, Any]],
    model: str,
    system_prompt: str
) -> Optional[str]:
    """Call the OpenRouter API to access various LLM models."""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "http://localhost:3000",
            "Content-Type": "application/json",
            "X-Title": "AI Conversation"  # Adding title for OpenRouter tracking
        }
        
        # Format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        for msg in conversation_history:
            if msg["role"] != "system":  # Skip system prompts
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,  # Using the exact model name from config
            "messages": messages,
            "temperature": API_TEMPERATURE,
            "max_tokens": API_MAX_TOKENS,
            "stream": False
        }
        
        print(f"\nSending to OpenRouter:")
        print(f"Model: {model}")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response data: {json.dumps(response_data, indent=2)}")
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message = response_data['choices'][0].get('message', {})
                if message and 'content' in message:
                    return message['content']
                else:
                    print(f"Unexpected message structure: {message}")
                    return None
            else:
                print(f"Unexpected response structure: {response_data}")
                return None
        else:
            error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
            print(error_msg)
            if response.status_code == 404:
                print("Model not found. Please check if the model name is correct.")
            elif response.status_code == 401:
                print("Authentication error. Please check your API key.")
            return f"Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return f"Error: Network error - {str(e)}"
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        print(f"Error type: {type(e)}")
        return f"Error: {str(e)}"

def call_replicate_api(
    prompt: str,
    conversation_history: List[Dict[str, Any]],
    model: str,
    gui: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    try:
        # Only use the prompt, ignore conversation history
        input_params = {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "prompt": prompt
        }
        
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input_params
        )
        
        image_url = str(output)
        
        # Save the image locally
        image_dir = Path("images")
        image_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = image_dir / f"generated_{timestamp}.jpg"
        
        response = requests.get(image_url)
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        if gui:
            gui.display_image(image_url)
        
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I have generated an image based on your prompt."
                }
            ],
            "prompt": prompt,
            "image_url": image_url,
            "image_path": str(image_path)
        }

    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error calling Flux: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading image from Flux: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error calling Flux API: {e}")
        return None

def call_deepseek_api(
    prompt: str,
    conversation_history: List[Dict[str, Any]],
    model: str,
    system_prompt: str
) -> Optional[Dict[str, Any]]:
    """Call the DeepSeek model through Replicate API."""
    try:
        # Format messages for the conversation history
        formatted_history = ""
        if system_prompt:
            formatted_history += f"System: {system_prompt}\n"
        
        # Add conversation history ensuring proper interleaving
        last_role = None
        for msg in conversation_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Skip if same role as last message
                if role == last_role:
                    continue
                    
                if isinstance(content, str):
                    formatted_history += f"{role.capitalize()}: {content}\n"
                    last_role = role
        
        # Add current prompt
        formatted_history += f"User: {prompt}\n"
        
        print(f"\nSending to DeepSeek via Replicate:")
        print(f"Formatted History:\n{formatted_history}")
        
        # Make API call
        output = replicate.run(
            "deepseek-ai/deepseek-r1",
            input={
                "prompt": formatted_history,
                "max_tokens": DEEPSEEK_MAX_TOKENS,
                "temperature": DEEPSEEK_TEMPERATURE
            }
        )
        
        # Collect the response - ensure we get the full output
        response_text = ""
        if isinstance(output, list):
            # Join all chunks to get the complete response
            response_text = ''.join(output)
        else:
            # If output is not a list, convert to string
            response_text = str(output)
            
        print(f"\nRaw Response: {response_text}")
        
        # Check for HTML contribution marker
        html_contribution = None
        conversation_part = response_text
        
        # Use more lenient pattern matching - just look for "HTML CONTRIBUTION" anywhere
        import re
        html_contribution_match = re.search(r'(?i)[-_\s]*HTML\s*CONTRIBUTION[-_\s]*', response_text)
        if html_contribution_match:
            parts = re.split(r'(?i)[-_\s]*HTML\s*CONTRIBUTION[-_\s]*', response_text, 1)
            if len(parts) > 1:
                conversation_part = parts[0].strip()
                html_contribution = parts[1].strip()
                print(f"Found HTML contribution with lenient pattern: {html_contribution[:100]}...")
        
        # Initialize result with content
        result = {
            "content": conversation_part
        }
        
        # Only extract and format chain of thought if SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT is True
        from config import SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT
        if SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT:
            # Try to extract reasoning from <think> tags in content
            reasoning = None
            content = conversation_part
            
            if content:
                # Try both <think> and <thinking> tags
                think_match = re.search(r'<(think|thinking)>(.*?)</\1>', content, re.DOTALL | re.IGNORECASE)
                if think_match:
                    reasoning = think_match.group(2).strip()
                    # Remove the thinking section from the content
                    content = re.sub(r'<(think|thinking)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # Format the response with both CoT and final answer
            display_text = ""
            if reasoning:
                display_text += f"[Chain of Thought]\n{reasoning}\n\n"
            if content:
                display_text += f"[Final Answer]\n{content}"
            
            # Add display field to result
            result["display"] = display_text
            # Update content to be the cleaned version without thinking tags
            result["content"] = content
        else:
            # If not showing chain of thought, just use the raw content
            # Still clean up any thinking tags from the content
            content = conversation_part
            if content:
                # Remove any thinking tags from the content
                content = re.sub(r'<(think|thinking)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
                result["content"] = content
        
        # Add HTML contribution if found
        if html_contribution:
            result["html_contribution"] = html_contribution
            
        return result

    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error calling DeepSeek: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error calling DeepSeek via Replicate: {e}")
        return None

def setup_image_directory() -> Path:
    """Create an 'images' directory in the project root if it doesn't exist"""
    image_dir = Path("images")
    image_dir.mkdir(exist_ok=True)
    return image_dir

def cleanup_old_images(image_dir: Path, max_age_hours: int = 24) -> None:
    """Remove images older than max_age_hours"""
    current_time = datetime.now()
    for image_file in image_dir.glob("*.jpg"):
        file_age = datetime.fromtimestamp(image_file.stat().st_mtime)
        if (current_time - file_age).total_seconds() > max_age_hours * 3600:
            image_file.unlink()

def load_ai_memory(ai_number: int) -> List[Dict[str, str]]:
    """Load AI conversation memory from JSON files"""
    try:
        memory_path = f"memory/ai{ai_number}/conversations.json"
        with open(memory_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
            # Ensure we're working with the array part
            if isinstance(conversations, dict) and "memories" in conversations:
                conversations = conversations["memories"]
        return conversations
    except Exception as e:
        print(f"Error loading AI{ai_number} memory: {e}")
        return []

def create_memory_prompt(conversations: List[Dict[str, str]]) -> str:
    """Convert memory JSON into conversation examples"""
    if not conversations:
        return ""
    
    prompt = "Previous conversations that demonstrate your personality:\n\n"
    
    # Add example conversations
    for convo in conversations:
        prompt += f"Human: {convo['human']}\n"
        prompt += f"Assistant: {convo['assistant']}\n\n"
    
    prompt += "Maintain this conversation style in your responses."
    return prompt 


def print_conversation_state(conversation: List[Dict[str, Any]]) -> None:
    """Print current conversation state for debugging"""
    print("Current conversation state:")
    for message in conversation:
        print(f"{message['role']}: {message['content'][:50]}...")  # Print first 50 characters of each message

def call_claude_vision_api(image_url: str) -> Optional[str]:
    """Have Claude analyze the generated image"""
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
        logger.exception(f"Error in Claude vision analysis: {e}")
        return None

def list_together_models() -> None:
    """List available Together AI models"""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHERAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.together.xyz/v1/models",
            headers=headers
        )
        
        print("\nAvailable Together AI Models:")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(json.dumps(models, indent=2))
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")

def start_together_model(model_id: str) -> bool:
    """Start a Together AI model"""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHERAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # URL encode the model ID
        encoded_model = requests.utils.quote(model_id, safe='')
        start_url = f"https://api.together.xyz/v1/models/{encoded_model}/start"
        
        print(f"\nAttempting to start model: {model_id}")
        print(f"Using URL: {start_url}")
        response = requests.post(
            start_url,
            headers=headers
        )
        
        print(f"Start request status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Model start request successful")
            return True
        else:
            print("Failed to start model")
            return False
            
    except Exception as e:
        print(f"Error starting model: {str(e)}")
        return False

def call_together_api(
    prompt: str,
    conversation_history: List[Dict[str, Any]],
    model: str,
    system_prompt: str
) -> Optional[str]:
    """Call the Together AI API"""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHERAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": TOGETHER_MAX_TOKENS,
            "temperature": TOGETHER_TEMPERATURE,
            "top_p": TOGETHER_TOP_P,
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            print(f"Together API Error Status: {response.status_code}")
            print(f"Response Body: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Together API: {str(e)}")
        return None

def open_html_in_browser(file_path: str = "conversation_full.html") -> None:
    """Open HTML file in default browser"""
    import webbrowser, os
    full_path = os.path.abspath(file_path)
    webbrowser.open('file://' + full_path)

def generate_image_from_text(text: str, model: str = "gpt-image-1") -> Dict[str, Any]:
    """Generate an image based on text using OpenAI's image generation API"""
    try:
        # Create a directory for the images if it doesn't exist
        image_dir = Path("images")
        image_dir.mkdir(exist_ok=True)
        
        # Create a timestamp for the image filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate the image
        result = openai_client.images.generate(
            model=model,
            prompt=text,
            n=1,  # Generate 1 image
        )
        
        # Get the base64 encoded image
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        # Save the image to a file
        image_path = image_dir / f"generated_{timestamp}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        print(f"Generated image saved to {image_path}")
        
        return {
            "success": True,
            "image_path": str(image_path),
            "timestamp": timestamp
        }
    except openai.APIError as e:
        logger.error(f"OpenAI API error generating image: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        logger.exception(f"Unexpected error generating image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# -------------------- Sora Video Utilities --------------------
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

    Returns a dict with keys: success, video_id, status, video_path (when completed), error
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

