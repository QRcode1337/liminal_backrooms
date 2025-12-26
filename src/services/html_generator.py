import os
from datetime import datetime
from src.ui.colors import COLORS

def update_conversation_html(conversation, filename="conversation_full.html"):
    """Update the full conversation HTML document with all messages"""
    try:
        # Generate HTML content for the conversation
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Liminal Backrooms</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: {COLORS['bg_dark']};
            --bg-panel: {COLORS['bg_medium']};
            --bg-message: {COLORS['bg_light']};
            --border-glow: {COLORS['border_glow']};
            --text-primary: {COLORS['text_normal']};
            --text-dim: {COLORS['text_dim']};
            --accent-cyan: {COLORS['accent_cyan']};
            --accent-purple: {COLORS['accent_purple']};
            --accent-blue: {COLORS['accent_blue']};
            --accent-orange: {COLORS['accent_orange']};
            --accent-pink: {COLORS['accent_pink']};
        }}

        * {{ box-sizing: border-box; }}

        body {{
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.7;
            color: var(--text-primary);
            background: var(--bg-dark);
            background-image:
                radial-gradient(ellipse at top, rgba(0, 255, 208, 0.03) 0%, transparent 50%),
                radial-gradient(ellipse at bottom, rgba(179, 136, 255, 0.03) 0%, transparent 50%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }}

        header {{
            text-align: center;
            margin-bottom: 50px;
            padding: 40px 20px;
            background: linear-gradient(135deg, rgba(0, 255, 208, 0.05) 0%, rgba(179, 136, 255, 0.05) 100%);
            border: 1px solid var(--border-glow);
            border-radius: 16px;
            position: relative;
            overflow: hidden;
        }}

        header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
        }}

        h1 {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-cyan);
            font-size: 2.2em;
            margin: 0 0 10px 0;
            font-weight: 500;
            letter-spacing: 2px;
            text-transform: uppercase;
            text-shadow: 0 0 30px rgba(0, 255, 208, 0.3);
        }}

        .subtitle {{
            color: var(--text-dim);
            font-size: 0.95em;
            font-weight: 300;
            letter-spacing: 1px;
        }}

        .message {{
            margin-bottom: 30px;
            padding: 24px;
            border-radius: 12px;
            background: var(--bg-message);
            border: 1px solid var(--border-glow);
            position: relative;
            transition: all 0.2s ease;
        }}

        .message:hover {{
            border-color: rgba(0, 255, 208, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}

        .message-content {{
            width: 100%;
        }}

        .message-image {{
            width: 100%;
            margin-top: 20px;
        }}

        .message-image img {{
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border-glow);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }}

        .user {{
            border-left: 3px solid var(--accent-cyan);
        }}

        .assistant {{
            border-left: 3px solid var(--accent-purple);
        }}

        .system {{
            background: rgba(255, 171, 64, 0.05);
            border-left: 3px solid var(--accent-orange);
            font-style: italic;
        }}

        .agent-notification {{
            background: linear-gradient(135deg, rgba(0, 255, 208, 0.08) 0%, rgba(79, 195, 247, 0.08) 100%);
            border: 1px solid rgba(0, 255, 208, 0.2);
            border-left: 3px solid var(--accent-cyan);
            padding: 16px 20px;
            margin: 16px 0;
            font-size: 0.9em;
            border-radius: 8px;
        }}

        .header {{
            font-weight: 500;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .ai-name {{
            color: var(--accent-purple);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95em;
        }}

        .model-name {{
            color: var(--text-dim);
            font-size: 0.85em;
            font-weight: 400;
        }}

        .user .ai-name {{
            color: var(--accent-cyan);
        }}

        .timestamp {{
            font-size: 0.75em;
            color: var(--text-dim);
            font-weight: 300;
        }}

        .content {{
            white-space: pre-wrap;
            font-size: 0.95em;
            line-height: 1.8;
        }}

        .greentext {{
            color: #789922;
            font-family: 'JetBrains Mono', monospace;
        }}

        p {{ margin: 0.6em 0; }}

        code {{
            background: rgba(0, 255, 208, 0.1);
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
            color: var(--accent-cyan);
        }}

        pre {{
            background: var(--bg-dark);
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
            margin: 20px 0;
            border: 1px solid var(--border-glow);
            color: var(--text-primary);
        }}

        footer {{
            margin-top: 60px;
            text-align: center;
            padding: 30px 20px;
            border-top: 1px solid var(--border-glow);
        }}

        footer p {{
            color: var(--text-dim);
            font-size: 0.85em;
            letter-spacing: 1px;
        }}

        footer a {{
            color: var(--accent-cyan);
            text-decoration: none;
        }}

        /* Share button */
        .share-bar {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }}

        .share-btn {{
            background: var(--bg-panel);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
            transition: all 0.2s ease;
        }}

        .share-btn:hover {{
            background: rgba(0, 255, 208, 0.1);
            box-shadow: 0 0 20px rgba(0, 255, 208, 0.2);
        }}

        @media (max-width: 600px) {{
            .container {{ padding: 20px 12px; }}
            h1 {{ font-size: 1.5em; }}
            .message {{ padding: 16px; }}
            .header {{ flex-direction: column; align-items: flex-start; }}
        }}
    </style>
</head>
<body>
    <div class="share-bar">
        <button class="share-btn" onclick="copyPageUrl()">📋 Copy Link</button>
    </div>

    <div class="container">
        <header>
            <h1>⟨ Liminal Backrooms ⟩</h1>
            <p class="subtitle">AI Conversation Archive</p>
        </header>

        <div id="conversation">"""

        # Add each message to the HTML content
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            ai_name = msg.get("ai_name", "")
            model = msg.get("model", "")
            timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

            # Skip special system messages or empty messages
            if role == "system" and msg.get("_type") == "branch_indicator":
                continue

            # Check if content is empty (handle both string and list)
            is_empty = False
            if isinstance(content, str):
                is_empty = not content.strip()
            elif isinstance(content, list):
                # For structured content, check if all text parts are empty
                text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                is_empty = not any(text_parts) and not any(part.get('type') == 'image' for part in content)
            else:
                is_empty = not content

            if is_empty:
                continue

            # Extract text content from structured messages
            text_content = ""
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                text_content = '\\n'.join(text_parts)

            # Process content to properly format code blocks and add greentext styling
            from html import escape
            processed_content = escape(text_content) if text_content else ""

            # Message class based on role and type
            message_class = role
            if msg.get("_type") == "agent_notification":
                message_class = "agent-notification"

            # Check if this message has an associated image
            has_image = False
            image_path = None
            image_base64 = None

            # Check for generated image path
            if hasattr(msg, "get") and callable(msg.get):
                image_path = msg.get("generated_image_path", None)
                if image_path:
                    has_image = True

            # Check for uploaded image in structured content
            if isinstance(content, list):
                for part in content:
                    if part.get('type') == 'image':
                        source = part.get('source', {})
                        if source.get('type') == 'base64':
                            image_base64 = source.get('data', '')
                            has_image = True
                            break

            # Start message div
            html_content += f'\\n        <div class="message {message_class}">'

            # Open content div
            html_content += f'\\n            <div class="message-content">'

            # Add header for assistant messages
            if role == "assistant":
                html_content += f'\\n                <div class="header"><span class="ai-name">{ai_name}</span>'
                if model:
                    html_content += f' <span class="model-name">({model})</span>'
                html_content += f' <span class="timestamp">{timestamp}</span></div>'
            elif role == "user":
                html_content += f'\\n                <div class="header"><span class="ai-name">User</span> <span class="timestamp">{timestamp}</span></div>'

            # Add message content
            html_content += f'\\n                <div class="content">{processed_content}</div>'

            # Close content div
            html_content += '\\n            </div>'

            # Add image if present - full width
            if has_image:
                html_content += f'\\n            <div class="message-image">'
                if image_base64:
                    # Use base64 data directly
                    html_content += f'\\n                <img src="data:image/jpeg;base64,{image_base64}" alt="Generated image" loading="lazy" />'
                elif image_path:
                    # Convert Windows path format to web format if needed
                    web_path = image_path.replace('\\\\', '/')
                    html_content += f'\\n                <img src="{web_path}" alt="Generated image" loading="lazy" />'
                html_content += f'\\n            </div>'

            # Close message div
            html_content += '\\n        </div>'

        # Close HTML document
        html_content += """
        </div>

        <footer>
            <p>Generated by <a href="#">Liminal Backrooms</a></p>
        </footer>
    </div>

    <script>
        function copyPageUrl() {
            const url = window.location.href;
            navigator.clipboard.writeText(url).then(() => {
                const btn = document.querySelector('.share-btn');
                btn.textContent = '✓ Copied!';
                setTimeout(() => { btn.textContent = '📋 Copy Link'; }, 2000);
            }).catch(() => {
                // Fallback for file:// URLs
                const text = document.documentElement.outerHTML;
                const blob = new Blob([text], {type: 'text/html'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'conversation.html';
                a.click();
                const btn = document.querySelector('.share-btn');
                btn.textContent = '✓ Downloaded!';
                setTimeout(() => { btn.textContent = '📋 Copy Link'; }, 2000);
            });
        }
    </script>
</body>
</html>"""

        # Write the HTML content to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Updated full conversation HTML document: {filename}")
        return True
    except Exception as e:
        print(f"Error updating conversation HTML: {e}")
        return False
