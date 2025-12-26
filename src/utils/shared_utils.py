import json
import webbrowser
import os
from pathlib import Path
from datetime import datetime

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

def read_shared_html(*args, **kwargs):
    return ""

def update_shared_html(*args, **kwargs):
    return False

def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        dict with keys: success, results (list of {title, url, snippet}), error
    """
    try:
        from ddgs import DDGS
    except ImportError:
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
        return {
            "success": False,
            "error": str(e)
        }
