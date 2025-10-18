# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

# Liminal Backrooms - Developer Guide

A PyQt6-based GUI application for dynamic, branching multi-AI conversations with visual network graph representation. Supports multiple AI providers (Claude, GPT, Gemini, Grok, DeepSeek, etc.) with forking and rabbitholing capabilities.

## Quickstart

**Prerequisites:**
- Python 3.10 or 3.11 (3.12 not supported)
- Poetry for dependency management
- API keys for desired providers

**Setup:**
```bash
poetry env use python3.11
poetry install
```

**Configure API keys** in `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
REPLICATE_API_TOKEN=your_key_here  # Optional
```

**Run:**
```bash
poetry run python main.py
```

## Development Commands

### Environment Management
```bash
# Install dependencies
poetry install

# Activate Poetry shell
poetry shell

# Reset environment if needed
poetry env remove --all
poetry env use python3.11
poetry install
```

### Running the Application
```bash
# Standard launch
poetry run python main.py

# With debug logging (if implemented)
export LOG_LEVEL=DEBUG
poetry run python main.py
```

### Code Quality
The project uses ruff for linting (configured in pyproject.toml):
```bash
# Check code
poetry run ruff check .

# Format code
poetry run ruff format .
```

### Testing
Currently no test suite exists. When tests are added:
```bash
poetry run pytest
```

## Core Architecture

### Module Structure

**main.py** - Application entry point and orchestration
- Creates QApplication and initializes PyQt6 GUI
- `ConversationManager` class coordinates conversation flow
- `Worker` class (QRunnable) executes AI turns asynchronously via QThreadPool
- Manages conversation state, branching, and HTML export
- Signal/slot architecture for async updates

**gui.py** - PyQt6 GUI components
- `LiminalBackroomsApp` - Main window with three-panel layout
- `NetworkGraphWidget` - Visual conversation graph with node positioning, edge animation, collision detection
- `ControlPanel` - Model selection, iterations, prompt style
- Custom context menus for forking/rabbitholing selected text
- Loading animations and conversation display

**config.py** - Centralized configuration
- `AI_MODELS` dict - Maps display names to model IDs
- `SYSTEM_PROMPT_PAIRS` dict - Predefined conversation styles
- Runtime settings: `TURN_DELAY`, `SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT`, `SHARE_CHAIN_OF_THOUGHT`

**shared_utils.py** - Provider API adapters
- `call_claude_api()` - Anthropic API
- `call_openai_api()` - OpenAI API
- `call_openrouter_api()` - OpenRouter multi-model access
- `call_replicate_api()` - Replicate (Flux image generation)
- `call_deepseek_api()` - DeepSeek via Replicate
- `generate_image_from_text()` - Image generation wrapper
- `open_html_in_browser()` - HTML conversation export

### Threading Model

- **Main Thread**: PyQt6 UI event loop
- **Worker Threads**: QThreadPool manages AI API calls via `Worker` (QRunnable)
- **Signals**: `WorkerSignals` class provides `finished`, `error`, `response`, `result`, `progress` signals
- Each AI turn spawns two workers (AI-1 and AI-2) that execute sequentially with configurable delay

### Conversation Data Model

**Message Structure:**
```python
{
    "role": "user" | "assistant" | "system",
    "content": str,
    "ai_name": "AI-1" | "AI-2",
    "model": str,  # Display name from AI_MODELS
    "hidden": bool,  # Optional, for hidden prompts
    "_type": str,  # Optional, e.g., "branch_indicator"
    "generated_image_path": str  # Optional, for auto-generated images
}
```

**Conversation State:**
- `main_conversation`: Primary conversation list
- `branch_conversations`: Dict mapping branch_id to branch data
- `active_branch`: Currently active branch ID or None
- Branch data includes: `type` (rabbithole/fork), `selected_text`, `conversation`, `parent`

## Branching System

### Rabbitholing (🐇)
- **Purpose**: Deep dive into a specific concept
- **Behavior**: 
  - Copies full parent conversation context
  - First TWO AI responses use focused prompt: `"'{selected_text}'!!!"`
  - Subsequent responses revert to standard prompts
  - Adds branch indicator to conversation
- **Visual**: Green nodes in graph

### Forking (🍴)
- **Purpose**: Explore alternative continuation from a point
- **Behavior**:
  - Copies conversation UP TO selected text
  - Truncates message at selection point
  - First response uses fork-specific prompt
  - Subsequent responses use standard prompts
  - Hidden instruction message ("...") starts the fork
- **Visual**: Yellow nodes in graph

### Branch Implementation
1. User right-clicks text and selects branch type
2. `rabbithole_callback()` or `fork_callback()` in ConversationManager
3. New branch_id generated (e.g., `"rabbithole_1728555555.123"`)
4. Branch conversation created with context and branch indicator
5. Node added to NetworkGraphWidget
6. `process_branch_input()` initiates AI turns for the branch

## Configuration and Extension

### Adding New AI Models

1. **Update config.py**:
```python
AI_MODELS = {
    "Display Name": "provider/model-id",
    # ... existing models
}
```

2. **For new providers**, add API handler in shared_utils.py:
```python
def call_new_provider_api(prompt, messages, model_id, system_prompt):
    # Implement API call logic
    # Return string or dict with 'content' key
```

3. **Route in main.py** `ai_turn()` function based on model_id pattern

### Adding System Prompt Pairs

Edit `SYSTEM_PROMPT_PAIRS` in config.py:
```python
SYSTEM_PROMPT_PAIRS = {
    "Your Style Name": {
        "AI_1": "System prompt for first AI...",
        "AI_2": "System prompt for second AI..."
    }
}
```

### Configuration Options

**In config.py:**
- `TURN_DELAY` - Seconds between AI turns (default: 2)
- `SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT` - Display reasoning for DeepSeek (default: False)
- `SHARE_CHAIN_OF_THOUGHT` - Include reasoning in context for next AI (default: False)

**GUI Settings:**
- Iterations: Number of back-and-forth exchanges (1-100)
- Auto-image generation: Toggle automatic image creation from responses
- Model selection: Independent for AI-1 and AI-2

## HTML Export

Conversations auto-export to `conversation_full.html` with:
- Dark themed styling
- Message-by-message layout with timestamps
- Generated images displayed inline
- Greentext styling for lines starting with '>'
- Code block formatting

Updated after each turn via `update_conversation_html()` in ConversationManager.

## Special Features

### Chain of Thought (DeepSeek)
- Extracts reasoning from `<think>` or `<thinking>` tags
- Displays separately if `SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT = True`
- Format: `[Chain of Thought]\n{reasoning}\n\n[Final Answer]\n{content}`

### Image Generation
- Automatic via checkbox or explicit image model selection
- Uses OpenAI gpt-image-1 via `generate_image_from_text()`
- Images saved to `images/` directory with timestamps
- Linked to messages via `generated_image_path` field

### Network Graph
- Nodes draggable for custom layout
- Physics simulation with repulsion/attraction forces
- Growing edge animations
- Node colors by type: main (blue), rabbithole (green), fork (yellow)
- Click nodes to switch between branches

## Memory System (Optional)

Place JSON files in `memories/` directory:
- `memories/ai-1_memories.json`
- `memories/ai-2_memories.json`

Format: Array of prior conversation snippets to inform AI personality

## Troubleshooting

### Poetry Installation Issues
```bash
# If Pillow fails to install
poetry env use python3.11
poetry install

# If Python version mismatch
poetry env remove --all
poetry env use python3.11
poetry install
```

### GUI Not Launching
- Ensure PyQt6 is installed: `poetry show pyqt6`
- Check display environment on Linux: `echo $DISPLAY`
- Launch from terminal, not Finder (macOS env variable issue)

### API Errors
- Verify API keys in `.env` and loaded: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('ANTHROPIC_API_KEY'))"`
- Check model ID matches provider expectations
- Monitor console output for detailed error messages
- Some models require specific API key (e.g., DeepSeek via Replicate needs REPLICATE_API_TOKEN)

### Branching Issues
- If duplicate messages appear, check conversation filtering logic in `ai_turn()`
- If images disappear on branch, this is a known GUI limitation - check `images/` folder
- Graph nodes overlapping: drag apart or disable physics with `apply_physics = False`

### Performance
- High memory usage: Limit conversation history or iterations
- Slow rendering: Reduce node count in graph or disable edge animations
- API timeouts: Reduce `max_tokens` in provider call functions

## Development Tips

- **Debugging conversation flow**: Add print statements in `ai_turn()` to trace message filtering
- **Testing new providers**: Start with simple single-turn conversations before branching
- **UI customization**: Colors defined in `COLORS` dict at top of gui.py
- **Async debugging**: Check Worker signals connect properly, use `finished.connect()` for cleanup
- **Branch logic**: Key distinction is in system prompt override for first 1-2 responses

## Project Context

**Inspired by**: Andy Ayrey and Janus  
**License**: MIT  
**Dependencies**: PyQt6, requests, replicate, python-dotenv, Pillow, anthropic, openai, together

**Note**: This is an experimental creative tool. Conversation quality depends heavily on prompt engineering and model selection.
