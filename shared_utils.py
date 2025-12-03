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
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not found. Please install it with 'pip install beautifulsoup4'")

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    print("ddgs not found. Install with: pip install ddgs")

# Load environment variables
load_dotenv()

# Initialize Anthropic client with API key
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def call_claude_api(prompt, messages, model_id, system_prompt=None, stream_callback=None, temperature=1.0):
    """Call the Claude API with the given messages and prompt
    
    Args:
        stream_callback: Optional function(chunk: str) to call with each streaming token
        temperature: Sampling temperature (0-2, default 1.0)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not found in environment variables"
    
    url = "https://api.anthropic.com/v1/messages"
    
    # Ensure we have a system prompt
    payload = {
        "model": model_id,
        "max_tokens": 4000,
        "temperature": temperature,
        "stream": stream_callback is not None  # Enable streaming if callback provided
    }
    
    # Set system if provided
    if system_prompt:
        payload["system"] = system_prompt
        print(f"CLAUDE API USING SYSTEM PROMPT: {system_prompt}")
    
    print(f"CLAUDE API USING TEMPERATURE: {temperature}")
    
    # Clean messages to remove duplicates
    filtered_messages = []
    seen_contents = set()
    
    for msg in messages:
        # Skip system messages (handled separately)
        if msg.get("role") == "system":
            continue
            
        # Get content - handle both string and list formats
        content = msg.get("content", "")
        
        # For duplicate detection, use a hashable representation (always a string)
        if isinstance(content, list):
            # For image messages, create a hash based on text content only
            text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
            content_hash = ''.join(text_parts)
        elif isinstance(content, str):
            content_hash = content
        else:
            # For any other type, convert to string
            content_hash = str(content) if content else ""
            
        # Check for duplicates
        if content_hash and content_hash in seen_contents:
            print(f"Skipping duplicate message in API call: {str(content_hash)[:30]}...")
            continue
            
        if content_hash:
            seen_contents.add(content_hash)
        filtered_messages.append(msg)
    
    # Add the current prompt as the final user message (if it's not already an image message)
    if prompt and not any(isinstance(msg.get("content"), list) for msg in filtered_messages[-1:]):
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
        if stream_callback:
            # Streaming mode using REST API directly
            payload["stream"] = True
            full_response = ""
            
            response = requests.post(url, json=payload, headers=headers, stream=True)
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_str = line_text[6:]  # Remove 'data: ' prefix
                            # Skip if this is a ping or message_stop event
                            if json_str.strip() in ['[DONE]', '']:
                                continue
                            try:
                                chunk_data = json.loads(json_str)
                                # Handle different event types from Claude's SSE stream
                                event_type = chunk_data.get('type')
                                
                                if event_type == 'content_block_delta':
                                    delta = chunk_data.get('delta', {})
                                    if delta.get('type') == 'text_delta':
                                        text = delta.get('text', '')
                                        if text:
                                            full_response += text
                                            stream_callback(text)
                            except json.JSONDecodeError:
                                continue
                return full_response
            else:
                return f"Error: API returned status {response.status_code}: {response.text}"
        else:
            # Non-streaming mode (original behavior)
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if 'content' in data and len(data['content']) > 0:
                for content_item in data['content']:
                    if content_item.get('type') == 'text':
                        return content_item.get('text', '')
                # Fallback if no text type content is found
                return str(data['content'])
            return "No content in response"
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

def call_llama_api(prompt, conversation_history, model, system_prompt):
    # Only use the last 3 exchanges to prevent context length issues
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
    
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
                "max_tokens": 3000,
                "temperature": 1.1,
                "top_p": 0.99,
                "repetition_penalty": 1.0
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
    except Exception as e:
        print(f"Error calling LLaMA API: {e}")
        return None

def call_openai_api(prompt, conversation_history, model, system_prompt):
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
            # Increase max_tokens and add n parameter
            max_tokens=4000,
            n=1,
            temperature=1,
            stream=True
        )
        
        collected_messages = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:  # Changed condition
                collected_messages.append(chunk.choices[0].delta.content)
                
        full_reply = ''.join(collected_messages)
        return full_reply
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def call_openrouter_api(prompt, conversation_history, model, system_prompt, stream_callback=None, temperature=1.0):
    """Call the OpenRouter API to access various LLM models.
    
    Args:
        stream_callback: Optional function(chunk: str) to call with each streaming token
        temperature: Sampling temperature (0-2, default 1.0)
    """
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "http://localhost:3000",
            "Content-Type": "application/json",
            "X-Title": "AI Conversation"  # Adding title for OpenRouter tracking
        }
        
        # Normalize model ID for OpenRouter - add provider prefix if missing
        openrouter_model = model
        if model.startswith("claude-") and not model.startswith("anthropic/"):
            openrouter_model = f"anthropic/{model}"
            print(f"Normalized Claude model ID for OpenRouter: {model} -> {openrouter_model}")
        
        # Format messages - need to handle structured content with images
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        def convert_to_openai_format(content, include_images=True):
            """Convert Anthropic-style image format to OpenAI/OpenRouter format.
            
            Args:
                content: The message content (string or list)
                include_images: If False, strip image content and keep only text
            """
            if not isinstance(content, list):
                return content
            
            converted = []
            for part in content:
                if part.get('type') == 'text':
                    converted.append({"type": "text", "text": part.get('text', '')})
                elif part.get('type') == 'image':
                    if include_images:
                        # Convert Anthropic format to OpenAI format
                        source = part.get('source', {})
                        if source.get('type') == 'base64':
                            media_type = source.get('media_type', 'image/png')
                            data = source.get('data', '')
                            converted.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                }
                            })
                    # If not including images, we skip this part (text description is already there)
                elif part.get('type') == 'image_url':
                    if include_images:
                        # Already in OpenAI format
                        converted.append(part)
                else:
                    # Pass through unknown types
                    converted.append(part)
            
            # If we stripped images and only have one text element, simplify to string
            if not include_images and len(converted) == 1 and converted[0].get('type') == 'text':
                return converted[0]['text']
            elif not include_images and len(converted) == 0:
                return ""
            
            return converted
        
        def build_messages(include_images=True, max_images=5):
            """Build the messages list, optionally stripping images.
            
            Args:
                include_images: If False, strip ALL images
                max_images: Maximum number of images to include (from most recent). 
                           Older images are stripped but text is preserved.
            """
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            
            if include_images and max_images > 0:
                # First pass: identify which messages have images (by index)
                image_message_indices = []
                for i, msg in enumerate(conversation_history):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        has_image = any(
                            part.get('type') in ('image', 'image_url') 
                            for part in content if isinstance(part, dict)
                        )
                        if has_image:
                            image_message_indices.append(i)
                
                # Determine which indices should keep their images (last N)
                indices_to_keep_images = set(image_message_indices[-max_images:]) if image_message_indices else set()
                
                if len(image_message_indices) > max_images:
                    stripped_count = len(image_message_indices) - max_images
                    print(f"[Context] Stripping {stripped_count} older images, keeping last {max_images}")
                
                # Build messages with selective image inclusion
                for i, msg in enumerate(conversation_history):
                    if msg["role"] != "system":
                        keep_images = i in indices_to_keep_images
                        msgs.append({
                            "role": msg["role"],
                            "content": convert_to_openai_format(msg["content"], include_images=keep_images)
                        })
            else:
                # No images mode - strip all
                for msg in conversation_history:
                    if msg["role"] != "system":
                        msgs.append({
                            "role": msg["role"],
                            "content": convert_to_openai_format(msg["content"], include_images=False)
                        })
            
            # Also convert the prompt if it's structured content (always include images in current prompt)
            msgs.append({"role": "user", "content": convert_to_openai_format(prompt, include_images)})
            return msgs
        
        def make_api_call(include_images=True, max_images=5):
            """Make the API call, returns (success, result_or_error)"""
            msgs = build_messages(include_images=include_images, max_images=max_images)
            
            payload = {
                "model": openrouter_model,
                "messages": msgs,
                "temperature": temperature,  # Use AI's custom temperature
                "max_tokens": 4000,
                "stream": stream_callback is not None
            }
            
            print(f"\nSending to OpenRouter:")
            print(f"Model: {model}")
            print(f"Temperature: {temperature}")
            print(f"Include images: {include_images}")
            # Log message summary (avoid huge base64 dumps)
            for i, m in enumerate(msgs):
                content = m.get('content', '')
                if isinstance(content, list):
                    parts_summary = [p.get('type', 'unknown') for p in content]
                    print(f"  [{i}] {m.get('role')}: [structured: {parts_summary}]")
                else:
                    preview = str(content)[:80] + "..." if len(str(content)) > 80 else content
                    print(f"  [{i}] {m.get('role')}: {preview}")
            
            if stream_callback:
                # Streaming mode
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180,
                    stream=True
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    full_response = ""
                    chunk_count = 0
                    last_finish_reason = None
                    debug_chunks = []  # Store first few chunks for debugging
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                json_str = line_text[6:]
                                if json_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk_data = json.loads(json_str)
                                    # Store first 5 chunks for debugging
                                    if len(debug_chunks) < 5:
                                        debug_chunks.append(chunk_data)
                                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                        choice = chunk_data['choices'][0]
                                        delta = choice.get('delta', {})
                                        content = delta.get('content', '')
                                        last_finish_reason = choice.get('finish_reason')
                                        if content:
                                            full_response += content
                                            stream_callback(content)
                                        chunk_count += 1
                                except json.JSONDecodeError:
                                    continue
                    # Log if response is empty
                    if not full_response or not full_response.strip():
                        print(f"[OpenRouter STREAM] Empty response from {model}", flush=True)
                        print(f"[OpenRouter STREAM]   Chunks received: {chunk_count}", flush=True)
                        print(f"[OpenRouter STREAM]   Last finish_reason: {last_finish_reason}", flush=True)
                        print(f"[OpenRouter STREAM]   Response repr: {repr(full_response)}", flush=True)
                        # Print the actual chunk data for debugging
                        for i, chunk in enumerate(debug_chunks):
                            print(f"[OpenRouter STREAM]   Chunk {i}: {json.dumps(chunk)[:300]}", flush=True)
                    return True, full_response
                else:
                    return False, (response.status_code, response.text)
            else:
                # Non-streaming mode
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    # Debug: log full response structure for empty responses
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        choice = response_data['choices'][0]
                        message = choice.get('message', {})
                        content = message.get('content', '') if message else ''
                        if content and content.strip():
                            return True, content
                        else:
                            # Log detailed info about empty response (avoiding base64)
                            import sys
                            print(f"[OpenRouter] Empty content from model: {model}", flush=True)
                            print(f"[OpenRouter]   Choice keys: {list(choice.keys())}", flush=True)
                            print(f"[OpenRouter]   Message keys: {list(message.keys()) if message else 'None'}", flush=True)
                            print(f"[OpenRouter]   Finish reason: {choice.get('finish_reason', 'unknown')}", flush=True)
                            print(f"[OpenRouter]   Content type: {type(content).__name__}, len: {len(content) if content else 0}", flush=True)
                            print(f"[OpenRouter]   Content repr: {repr(content)}", flush=True)
                            # Check for refusal or other indicators
                            if message.get('refusal'):
                                print(f"[OpenRouter]   Refusal: {message.get('refusal')}", flush=True)
                            # Check for tool_calls that might indicate the model is doing something else
                            if message.get('tool_calls'):
                                print(f"[OpenRouter]   Tool calls: {len(message.get('tool_calls'))} call(s)", flush=True)
                            sys.stdout.flush()
                            return True, None
                    else:
                        print(f"[OpenRouter] No choices in response. Keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'non-dict'}")
                    return True, None
                else:
                    return False, (response.status_code, response.text)
        
        # Try with images first
        success, result = make_api_call(include_images=True)
        print(f"[OpenRouter] First call result - success: {success}, result type: {type(result).__name__}, result: {repr(result)[:100] if result else 'None'}", flush=True)
        
        if success:
            # Check for empty response and retry once
            if result is None or (isinstance(result, str) and not result.strip()):
                print(f"[OpenRouter] WARNING: Model {model} returned empty response, retrying...", flush=True)
                import time
                time.sleep(1)
                success, result = make_api_call(include_images=True)
                print(f"[OpenRouter] Retry result - success: {success}, result type: {type(result).__name__}, result: {repr(result)[:100] if result else 'None'}", flush=True)
                if success and result and (not isinstance(result, str) or result.strip()):
                    return result
                print(f"[OpenRouter] WARNING: Model {model} returned empty response again after retry", flush=True)
                return "[Model returned empty response - it may be experiencing issues]"
            return result
        
        # Check if error is due to model not supporting images
        status_code, error_text = result
        if status_code == 404 and "support image" in error_text.lower():
            print(f"[OpenRouter] Model {model} doesn't support images, retrying without images...")
            success, result = make_api_call(include_images=False)
            if success:
                return result
            status_code, error_text = result
        
        # Handle other errors
        error_msg = f"OpenRouter API error {status_code}: {error_text}"
        print(error_msg)
        if status_code == 404:
            print("Model not found or doesn't support this request type.")
        elif status_code == 401:
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

def call_replicate_api(prompt, conversation_history, model, gui=None):
    try:
        # Only use the prompt, ignore conversation history
        input_params = {
            "width": 1024,
            "height": 1024,
            "prompt": prompt
        }
        
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input_params
        )
        
        image_url = str(output)
        
        # Save the image locally (include microseconds to avoid collisions)
        image_dir = Path("images")
        image_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
        
    except Exception as e:
        print(f"Error calling Flux API: {e}")
        return None

def call_deepseek_api(prompt, conversation_history, model, system_prompt, stream_callback=None):
    """Call the DeepSeek model through OpenRouter API."""
    try:
        import re
        from config import SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT
        
        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})
        
        # Add current prompt if provided
        if prompt:
            messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        }
        
        payload = {
            "model": "deepseek/deepseek-r1",
            "messages": messages,
            "max_tokens": 8000,
            "temperature": 1,
            "stream": stream_callback is not None
        }
        
        print(f"\nSending to DeepSeek via OpenRouter:")
        print(f"Model: deepseek/deepseek-r1")
        print(f"Messages: {len(messages)} messages")
        
        if stream_callback:
            # Streaming mode
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_str = line_text[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(json_str)
                                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_response += content
                                        stream_callback(content)
                            except json.JSONDecodeError:
                                continue
                response_text = full_response
            else:
                error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
                print(error_msg)
                return None
        else:
            # Non-streaming mode
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data['choices'][0]['message']['content']
            else:
                error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
                print(error_msg)
                return None
        
        print(f"\nRaw Response: {response_text[:500]}...")
        
        # Initialize result with content
        result = {
            "content": response_text,
            "model": "deepseek/deepseek-r1"
        }
        
        # Extract and format chain of thought if enabled
        if SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT:
            reasoning = None
            content = response_text
            
            if content:
                # Try both <think> and <thinking> tags
                think_match = re.search(r'<(think|thinking)>(.*?)</\1>', content, re.DOTALL | re.IGNORECASE)
                if think_match:
                    reasoning = think_match.group(2).strip()
                    content = re.sub(r'<(think|thinking)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            
            display_text = ""
            if reasoning:
                display_text += f"[Chain of Thought]\n{reasoning}\n\n"
            if content:
                display_text += f"[Final Answer]\n{content}"
            
            result["display"] = display_text
            result["content"] = content
        else:
            # Clean up thinking tags from content
            content = response_text
            if content:
                content = re.sub(r'<(think|thinking)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
                result["content"] = content
        
        return result
        
    except Exception as e:
        print(f"Error calling DeepSeek via OpenRouter: {e}")
        print(f"Error type: {type(e)}")
        return None

def setup_image_directory():
    """Create an 'images' directory in the project root if it doesn't exist"""
    image_dir = Path("images")
    image_dir.mkdir(exist_ok=True)
    return image_dir

def cleanup_old_images(image_dir, max_age_hours=24):
    """Remove images older than max_age_hours"""
    current_time = datetime.now()
    for image_file in image_dir.glob("*.jpg"):
        file_age = datetime.fromtimestamp(image_file.stat().st_mtime)
        if (current_time - file_age).total_seconds() > max_age_hours * 3600:
            image_file.unlink()

def load_ai_memory(ai_number):
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

def create_memory_prompt(conversations):
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


def print_conversation_state(conversation):
    print("Current conversation state:")
    for message in conversation:
        content = message.get('content', '')
        # Safely preview content - handle both string and list (structured) content
        if isinstance(content, str):
            preview = content[:50] + "..." if len(content) > 50 else content
        else:
            preview = f"[structured content with {len(content)} parts]"
        print(f"{message['role']}: {preview}")

def call_claude_vision_api(image_url):
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
        print(f"Error in vision analysis: {e}")
        return None

def list_together_models():
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
            print(f"Error Response: {response.text[:500]}..." if len(response.text) > 500 else f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")

def start_together_model(model_id):
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
        print(f"Response: {response.text[:200]}..." if len(response.text) > 200 else f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Model start request successful")
            return True
        else:
            print("Failed to start model")
            return False
            
    except Exception as e:
        print(f"Error starting model: {str(e)}")
        return False

def call_together_api(prompt, conversation_history, model, system_prompt):
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
            "max_tokens": 500,
            "temperature": 0.9,
            "top_p": 0.95,
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
            print(f"Response Body: {response.text[:500]}..." if len(response.text) > 500 else f"Response Body: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Together API: {str(e)}")
        return None

def read_shared_html(*args, **kwargs):
    return ""

def update_shared_html(*args, **kwargs):
    return False

def open_html_in_browser(file_path="conversation_full.html"):
    import webbrowser, os
    full_path = os.path.abspath(file_path)
    webbrowser.open('file://' + full_path)

def create_initial_living_document(*args, **kwargs):
    return ""

def read_living_document(*args, **kwargs):
    return ""

def process_living_document_edits(result, model_name):
    return result

def generate_image_from_text(text, model="google/gemini-3-pro-image-preview"):
    """Generate an image based on text using OpenRouter's image generation API"""
    try:
        # Create a directory for the images if it doesn't exist
        image_dir = Path("images")
        image_dir.mkdir(exist_ok=True)
        
        # Create a timestamp for the image filename (include microseconds to avoid collisions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Call OpenRouter API for image generation
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
            "max_tokens": 1024  # Limit tokens for image generation to avoid credit issues
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
            
            # The generated image will be in the assistant message
            if result.get("choices"):
                message = result["choices"][0].get("message", {})
                
                # Check for images in the message
                if message.get("images"):
                    for image in message["images"]:
                        image_url = image["image_url"]["url"]  # Base64 data URL
                        print(f"Generated image URL (first 50 chars): {image_url[:50]}...")
                        
                        # Handle base64 data URL
                        if image_url.startswith('data:image'):
                            try:
                                # Detect actual image format from data URL header
                                # Format: data:image/jpeg;base64,... or data:image/png;base64,...
                                ext = ".jpg"  # Default to jpg
                                if image_url.startswith('data:image/png'):
                                    ext = ".png"
                                elif image_url.startswith('data:image/gif'):
                                    ext = ".gif"
                                elif image_url.startswith('data:image/webp'):
                                    ext = ".webp"
                                
                                # Extract base64 data after comma
                                base64_data = image_url.split(',', 1)[1] if ',' in image_url else image_url
                                
                                # Decode base64 to image
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
                            # If it's a regular URL, download it
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
                
                # No images in response
                print(f"No images in response. Message keys: {list(message.keys()) if isinstance(message, dict) else 'non-dict'}")
                return {
                    "success": False,
                    "error": "No images in API response"
                }
            else:
                print(f"No choices in response. Result keys: {list(result.keys()) if isinstance(result, dict) else 'non-dict'}")
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

# -------------------- Web Search Utilities --------------------
def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        dict with keys: success, results (list of {title, url, snippet}), error
    """
    if DDGS is None:
        return {
            "success": False,
            "error": "ddgs package not installed. Run: pip install ddgs"
        }
    
    try:
        print(f"[WebSearch] Searching for: {query}")
        
        # Use the new ddgs API - prioritize news for current events queries
        ddgs = DDGS()
        formatted_results = []
        
        # For queries about current events, use news search first
        is_news_query = any(term in query.lower() for term in ["news", "today", "latest", "2025", "drama", "announcement", "release"])
        
        if is_news_query:
            print(f"[WebSearch] Detected news query, searching news first...")
            try:
                news_results = list(ddgs.news(query, region="wt-wt", safesearch="off", max_results=max_results))
                for r in news_results:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", r.get("link", "")),
                        "snippet": r.get("body", r.get("excerpt", ""))
                    })
                print(f"[WebSearch] Found {len(formatted_results)} news results")
            except Exception as e:
                print(f"[WebSearch] News search failed: {e}")
        
        # If we don't have enough results, try text search
        if len(formatted_results) < max_results:
            remaining = max_results - len(formatted_results)
            try:
                text_results = list(ddgs.text(
                    query, 
                    region="us-en",  # Force US English results
                    safesearch="off",
                    max_results=remaining
                ))
                for r in text_results:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", ""))
                    })
                print(f"[WebSearch] Added {len(text_results)} text results, total: {len(formatted_results)}")
            except Exception as e:
                print(f"[WebSearch] Text search failed: {e}")
        
        return {
            "success": True,
            "results": formatted_results,
            "query": query
        }
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

