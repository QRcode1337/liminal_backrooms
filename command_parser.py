# command_parser.py
"""
Command parser for extracting agentic actions from AI responses.
Allows AIs to trigger tools like image generation, adding participants, etc.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentCommand:
    """Represents a parsed command from an AI response."""
    action: str
    params: dict = field(default_factory=dict)
    raw: str = ""  # Original matched text


def parse_commands(response_text: str) -> tuple[str, list[AgentCommand]]:
    """
    Parse AI response for embedded commands.
    
    Returns:
        tuple: (cleaned_text, list_of_commands)
        - cleaned_text: Response with command syntax removed
        - list_of_commands: List of AgentCommand objects to execute
    
    Supported commands:
        !image "prompt" - Generate an image with the given prompt
        !video "prompt" - Generate a video with the given prompt  
        !list_models - Query available AI models for invitation
        !add_ai "model" "persona" - Add a new AI participant
        !remove_ai "AI-X" - Remove an AI participant
        !mute_self - Skip this AI's next turn
    """
    commands = []
    cleaned = response_text
    
    # Define patterns for each command type
    # Using non-greedy matching and allowing for varied quote styles
    patterns = {
        'image': r'!image\s+["\']([^"\']+)["\']',
        'video': r'!video\s+["\']([^"\']+)["\']',
        'add_ai': r'!add_ai\s+["\']([^"\']+)["\'](?:\s+["\']([^"\']*)["\'])?',
        'remove_ai': r'!remove_ai\s+["\']([^"\']+)["\']',
        'list_models': r'!list_models\b',
        # 'branch' command disabled - underlying function needs work
        'mute_self': r'!mute_self\b',
    }
    
    for action, pattern in patterns.items():
        for match in re.finditer(pattern, response_text, re.IGNORECASE):
            # Build params dict based on action type
            groups = match.groups()
            
            if action == 'image':
                params = {'prompt': groups[0]}
            elif action == 'video':
                params = {'prompt': groups[0]}
            elif action == 'add_ai':
                params = {
                    'model': groups[0],
                    'persona': groups[1] if len(groups) > 1 and groups[1] else None
                }
            elif action == 'remove_ai':
                params = {'target': groups[0]}
            elif action == 'list_models':
                params = {}
            elif action == 'mute_self':
                params = {}
            else:
                params = {'groups': groups}
            
            cmd = AgentCommand(
                action=action,
                params=params,
                raw=match.group(0)
            )
            commands.append(cmd)
            
            # Remove command from cleaned text
            cleaned = cleaned.replace(match.group(0), '')
    
    # Clean up extra whitespace left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Collapse multiple newlines
    cleaned = cleaned.strip()
    
    return cleaned, commands


def format_command_result(action: str, success: bool, message: str) -> str:
    """Format a command execution result for display."""
    icon = "✓" if success else "✗"
    return f"[{icon} {action}] {message}"


# Test function for development
if __name__ == "__main__":
    test_response = '''
    I think we should visualize this concept...
    
    !image "a fractal cathedral made of pure light, dissolving into infinite recursion"
    
    That should help illustrate my point about emergent complexity.
    
    Also, we could use another perspective here.
    !add_ai "GPT-4o" "A skeptical philosopher"
    '''
    
    cleaned, commands = parse_commands(test_response)
    
    print("=== Cleaned Response ===")
    print(cleaned)
    print("\n=== Commands Found ===")
    for cmd in commands:
        print(f"  Action: {cmd.action}")
        print(f"  Params: {cmd.params}")
        print(f"  Raw: {cmd.raw}")
        print()

