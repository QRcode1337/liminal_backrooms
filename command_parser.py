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
        !search "query" - Search the web and share results with the group
        !fetch "url" - Read a web page and share an excerpt with the group
        !prompt "text" - Append text to this AI's own system prompt
        !list_models - Query available AI models for invitation
        !add_ai "model" "persona" - Add a new AI participant
        !remove_ai "AI-X" - Remove an AI participant
        !mute_self - Skip this AI's next turn
        !vote "question" [option1, option2, ...] - Start a poll with optional choices
        !whisper "target" "message" - Private message to a slot (AI-4), model id, or name
    """
    commands = []
    cleaned = response_text
    
    # Define patterns for each command type
    # Using patterns that match opening quote to same closing quote
    # Double-quoted strings can contain single quotes and vice versa
    patterns = {
        # Match "..." (can contain ') or '...' (can contain ")
        'image': r'!image\s+(?:"([^"]+)"|\'([^\']+)\')',
        'video': r'!video\s+(?:"([^"]+)"|\'([^\']+)\')',
        'search': r'!search\s+(?:"([^"]+)"|\'([^\']+)\')',
        'fetch': r'!fetch\s+(?:"([^"]+)"|\'([^\']+)\'|(https?://[^\s"\'<>]+))',
        'prompt': r'!prompt\s+(?:"([^"]+)"|\'([^\']+)\')',
        'temperature': r'!temperature\s+([\d.]+)',  # Match decimal number like 0.7, 1.5, etc.
        'add_ai': r'!add_ai\s+(?:"([^"]+)"|\'([^\']+)\')(?:\s+(?:"([^"]*)"|\'([^\']*)\'))?',
        'remove_ai': r'!remove_ai\s+(?:"([^"]+)"|\'([^\']+)\')',
        'list_models': r'!list_models\b',
        # 'branch' command disabled - underlying function needs work
        'mute_self': r'!mute_self\b',
        'vote': r'!vote\s+(?:"([^"]+)"|\'([^\']+)\')\s*(?:\[([^\]]*)\])?',
        'whisper': r'!whisper\s+(?:"([^"]+)"|\'([^\']+)\'|([^\s"\']+))\s+(?:"([^"]+)"|\'([^\']+)\')',
    }
    
    for action, pattern in patterns.items():
        for match in re.finditer(pattern, response_text, re.IGNORECASE):
            # Build params dict based on action type
            groups = match.groups()
            
            # Helper to get first non-None group (handles alternation patterns)
            def get_first_value(*indices):
                for i in indices:
                    if i < len(groups) and groups[i] is not None:
                        return groups[i]
                return None
            
            if action == 'image':
                # Groups 0 or 1 (double or single quoted)
                params = {'prompt': get_first_value(0, 1)}
            elif action == 'video':
                # Groups 0 or 1 (double or single quoted)
                params = {'prompt': get_first_value(0, 1)}
            elif action == 'search':
                # Groups 0 or 1 (double or single quoted)
                params = {'query': get_first_value(0, 1)}
            elif action == 'fetch':
                # Quoted (groups 0/1) or bare URL (group 2)
                params = {'url': get_first_value(0, 1, 2)}
            elif action == 'prompt':
                # Groups 0 or 1 (double or single quoted)
                params = {'text': get_first_value(0, 1)}
            elif action == 'temperature':
                # Single group - the decimal number
                params = {'value': groups[0] if groups else None}
            elif action == 'add_ai':
                # Model: groups 0 or 1, Persona: groups 2 or 3
                params = {
                    'model': get_first_value(0, 1),
                    'persona': get_first_value(2, 3)
                }
            elif action == 'remove_ai':
                params = {'target': get_first_value(0, 1)}
            elif action == 'list_models':
                params = {}
            elif action == 'vote':
                params = {
                    'question': get_first_value(0, 1),
                    'options': get_first_value(2, None)  # Optional comma-separated options
                }
            elif action == 'whisper':
                params = {
                    'target': get_first_value(0, 1, 2),
                    'message': get_first_value(3, 4)
                }
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
            
            # Strip !prompt and !temperature commands from text so other AIs don't see them
            # (keeps self-modifications private to each AI)
            if action in ('prompt', 'temperature', 'whisper'):
                cleaned = cleaned.replace(match.group(0), '')
    
    # Clean up extra whitespace but preserve content
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Collapse multiple newlines
    cleaned = cleaned.strip()
    
    return cleaned, commands


def _whisper_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def resolve_participant_target(target: str, roster: list) -> tuple[Optional[str], Optional[str]]:
    """Map a whisper target to an active AI slot.

    roster items are (ai_name, model_id, display_name), e.g.
    ("AI-4", "xai/grok-4.20-0309-non-reasoning", "Grok 4.20").
    """
    if not target or not str(target).strip():
        return None, "missing target"

    raw = str(target).strip()
    names = [name for name, _, _ in roster]

    slot_match = re.fullmatch(r"(?:AI-)?(\d+)", raw, re.IGNORECASE)
    if slot_match:
        slot = f"AI-{int(slot_match.group(1))}"
        if slot in names:
            return slot, None
        active = ", ".join(names) if names else "none"
        return None, f"{slot} doesn't exist (only {active} active)"

    needle = _whisper_key(raw)
    if not needle:
        return None, f"invalid target '{target}'"

    def _fields(entry):
        name, model_id, display = entry
        model_id = model_id or ""
        display = display or ""
        leaf = model_id.split("/")[-1]
        return {
            "name": name,
            "model": model_id.lower(),
            "display": display.lower(),
            "model_n": _whisper_key(model_id),
            "display_n": _whisper_key(display),
            "leaf_n": _whisper_key(leaf),
        }

    def _ambiguous(matches):
        detail = ", ".join(
            f"{name} ({model_id})" for name, model_id, _ in matches
        )
        return None, f"ambiguous target '{target}' — {detail}"

    exact_model = [p for p in roster if (p[1] or "").lower() == raw.lower()]
    if len(exact_model) == 1:
        return exact_model[0][0], None
    if len(exact_model) > 1:
        return _ambiguous(exact_model)

    exact_display = [p for p in roster if (p[2] or "").lower() == raw.lower()]
    if len(exact_display) == 1:
        return exact_display[0][0], None
    if len(exact_display) > 1:
        return _ambiguous(exact_display)

    exact_norm = [
        p
        for p in roster
        if (
            (fields := _fields(p))
            and (
                fields["display_n"] == needle
                or fields["leaf_n"] == needle
                or fields["model_n"] == needle
            )
        )
    ]
    if len(exact_norm) == 1:
        return exact_norm[0][0], None
    if len(exact_norm) > 1:
        return _ambiguous(exact_norm)

    partial = [
        p
        for p in roster
        if needle in _fields(p)["model_n"] or needle in _fields(p)["display_n"]
    ]
    if len(partial) == 1:
        return partial[0][0], None
    if len(partial) > 1:
        return _ambiguous(partial)

    return None, f"invalid target '{target}'"


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



_MENTION_PATTERN = re.compile(r'(?<![\w@])@AI-([1-5])\b', re.IGNORECASE)


def extract_mentions(text: str) -> list[str]:
    """Return slot names (e.g. ["AI-3"]) mentioned as @AI-N, in order, de-duplicated."""
    seen = []
    for match in _MENTION_PATTERN.finditer(text or ""):
        slot = f"AI-{match.group(1)}"
        if slot not in seen:
            seen.append(slot)
    return seen
