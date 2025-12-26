import os
import requests
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Third-party imports for API clients
import replicate
import openai
from anthropic import Anthropic
from openai import OpenAI
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    print("ddgs not found. Install with: pip install ddgs")

# Load environment variables
load_dotenv()

# Initialize clients
anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Import config
from src.core.config import SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT

def call_claude_api(prompt, messages, model_id, system_prompt=None, stream_callback=None, temperature=1.0):
    """Call the Claude API with the given messages and prompt"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not found in environment variables"

    url = "https://api.anthropic.com/v1/messages"

    payload = {
        "model": model_id,
        "max_tokens": 4000,
        "temperature": temperature,
        "stream": stream_callback is not None
    }

    if system_prompt:
        payload["system"] = system_prompt
        print(f"CLAUDE API USING SYSTEM PROMPT: {system_prompt}")

    print(f"CLAUDE API USING TEMPERATURE: {temperature}")

    # Filter messages
    filtered_messages = []
    seen_contents = set()

    for msg in messages:
        if msg.get("role") == "system":
            continue

        content = msg.get("content", "")

        # Create hashable content
        if isinstance(content, list):
            text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
            content_hash = ''.join(text_parts)
        elif isinstance(content, str):
            content_hash = content
        else:
            content_hash = str(content) if content else ""

        if content_hash and content_hash in seen_contents:
            print(f"Skipping duplicate message in API call: {str(content_hash)[:30]}...")
            continue

        if content_hash:
            seen_contents.add(content_hash)
        filtered_messages.append(msg)

    # Add the current prompt as the final user message
    if prompt and not any(isinstance(msg.get("content"), list) for msg in filtered_messages[-1:]):
        filtered_messages.append({
            "role": "user",
            "content": prompt
        })

    payload["messages"] = filtered_messages

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    try:
        if stream_callback:
            payload["stream"] = True
            full_response = ""

            response = requests.post(url, json=payload, headers=headers, stream=True)

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_str = line_text[6:]
                            if json_str.strip() in ['[DONE]', '']:
                                continue
                            try:
                                chunk_data = json.loads(json_str)
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
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if 'content' in data and len(data['content']) > 0:
                for content_item in data['content']:
                    if content_item.get('type') == 'text':
                        return content_item.get('text', '')
                return str(data['content'])
            return "No content in response"
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

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
            max_tokens=4000,
            n=1,
            temperature=1,
            stream=True
        )

        collected_messages = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                collected_messages.append(chunk.choices[0].delta.content)

        full_reply = ''.join(collected_messages)
        return full_reply

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def call_openrouter_api(prompt, conversation_history, model, system_prompt, stream_callback=None, temperature=1.0):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "http://localhost:3000",
            "Content-Type": "application/json",
            "X-Title": "AI Conversation"
        }

        openrouter_model = model
        if model.startswith("claude-") and not model.startswith("anthropic/"):
            openrouter_model = f"anthropic/{model}"
            print(f"Normalized Claude model ID for OpenRouter: {model} -> {openrouter_model}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        def convert_to_openai_format(content, include_images=True):
            if not isinstance(content, list):
                return content

            converted = []
            for part in content:
                if part.get('type') == 'text':
                    converted.append({"type": "text", "text": part.get('text', '')})
                elif part.get('type') == 'image':
                    if include_images:
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
                elif part.get('type') == 'image_url':
                    if include_images:
                        converted.append(part)
                else:
                    converted.append(part)

            if not include_images and len(converted) == 1 and converted[0].get('type') == 'text':
                return converted[0]['text']
            elif not include_images and len(converted) == 0:
                return ""

            return converted

        def build_messages(include_images=True, max_images=5):
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})

            if include_images and max_images > 0:
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

                indices_to_keep_images = set(image_message_indices[-max_images:]) if image_message_indices else set()

                if len(image_message_indices) > max_images:
                    stripped_count = len(image_message_indices) - max_images
                    print(f"[Context] Stripping {stripped_count} older images, keeping last {max_images}")

                for i, msg in enumerate(conversation_history):
                    if msg["role"] != "system":
                        keep_images = i in indices_to_keep_images
                        msgs.append({
                            "role": msg["role"],
                            "content": convert_to_openai_format(msg["content"], include_images=keep_images)
                        })
            else:
                for msg in conversation_history:
                    if msg["role"] != "system":
                        msgs.append({
                            "role": msg["role"],
                            "content": convert_to_openai_format(msg["content"], include_images=False)
                        })

            msgs.append({"role": "user", "content": convert_to_openai_format(prompt, include_images)})
            return msgs

        def make_api_call(include_images=True, max_images=5):
            msgs = build_messages(include_images=include_images, max_images=max_images)

            payload = {
                "model": openrouter_model,
                "messages": msgs,
                "temperature": temperature,
                "max_tokens": 4000,
                "stream": stream_callback is not None
            }

            print(f"\nSending to OpenRouter:")
            print(f"Model: {model}")
            print(f"Temperature: {temperature}")

            if stream_callback:
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
                                        choice = chunk_data['choices'][0]
                                        delta = choice.get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            full_response += content
                                            stream_callback(content)
                                except json.JSONDecodeError:
                                    continue
                    return True, full_response
                else:
                    return False, (response.status_code, response.text)
            else:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        choice = response_data['choices'][0]
                        message = choice.get('message', {})
                        content = message.get('content', '') if message else ''
                        if content and content.strip():
                            return True, content
                        else:
                            return True, None
                    return True, None
                else:
                    return False, (response.status_code, response.text)

        success, result = make_api_call(include_images=True)

        if success:
            if result is None or (isinstance(result, str) and not result.strip()):
                print(f"[OpenRouter] WARNING: Model {model} returned empty response, retrying...", flush=True)
                import time
                time.sleep(1)
                success, result = make_api_call(include_images=True)
                if success and result and (not isinstance(result, str) or result.strip()):
                    return result
                return "[Model returned empty response - it may be experiencing issues]"
            return result

        status_code, error_text = result
        if status_code == 404 and "support image" in error_text.lower():
            print(f"[OpenRouter] Model {model} doesn't support images, retrying without images...")
            success, result = make_api_call(include_images=False)
            if success:
                return result
            status_code, error_text = result

        error_msg = f"OpenRouter API error {status_code}: {error_text}"
        print(error_msg)
        return f"Error: {error_msg}"

    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: Network error - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def call_deepseek_api(prompt, conversation_history, model, system_prompt, stream_callback=None):
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in conversation_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})

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

        if stream_callback:
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

        result = {
            "content": response_text,
            "model": "deepseek/deepseek-r1"
        }

        if SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT:
            reasoning = None
            content = response_text

            if content:
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
            content = response_text
            if content:
                content = re.sub(r'<(think|thinking)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
                result["content"] = content

        return result

    except Exception as e:
        print(f"Error calling DeepSeek via OpenRouter: {e}")
        return None

def call_replicate_api(prompt, conversation_history, model, gui=None):
    try:
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

def call_llama_api(prompt, conversation_history, model, system_prompt):
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
    formatted_history = ""
    for message in recent_history:
        if message["role"] == "user":
            formatted_history += f"Human: {message['content']}\n"
        else:
            formatted_history += f"Assistant: {message['content']}\n"
    formatted_history += f"Human: {prompt}\nAssistant:"

    try:
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
            stream=True
        ):
            if chunk is not None:
                response_chunks.append(chunk)

        response = ''.join(response_chunks)
        return response
    except Exception as e:
        print(f"Error calling LLaMA API: {e}")
        return None

def call_together_api(prompt, conversation_history, model, system_prompt):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHERAI_API_KEY')}",
            "Content-Type": "application/json"
        }

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
            return None

    except Exception as e:
        print(f"Error calling Together API: {str(e)}")
        return None
