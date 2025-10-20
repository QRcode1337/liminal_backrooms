# AGENTS.md - AI Assistant Guidelines

## Byterover MCP Tools

You are given two tools from Byterover MCP server, including

### 1. `byterover-store-knowledge`
You `MUST` always use this tool when:

+ Learning new patterns, APIs, or architectural decisions from the codebase
+ Encountering error solutions or debugging techniques
+ Finding reusable code patterns or utility functions
+ Completing any significant task or plan implementation

### 2. `byterover-retrieve-knowledge`
You `MUST` always use this tool when:

+ Starting any new task or implementation to gather relevant context
+ Before making architectural decisions to understand existing patterns
+ When debugging issues to check for previous solutions
+ Working with unfamiliar parts of the codebase

## Project Overview

**Liminal Backrooms** is a PyQt6-based GUI application for dynamic, branching multi-AI conversations with visual network graph representation. It supports multiple AI providers (Claude, GPT, Gemini, Grok, DeepSeek, etc.) with forking and rabbitholing capabilities.

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

## Development Setup

### Prerequisites
- Python 3.10 or 3.11 (3.12 not supported)
- Poetry for dependency management
- API keys for desired providers

### Environment Setup
```bash
poetry env use python3.11
poetry install
```

### API Configuration
Configure API keys in `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
REPLICATE_API_TOKEN=your_key_here  # Optional
```

### Running the Application
```bash
poetry run python main.py
```

## Common Issues & Solutions

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

### Signal/Threading Issues
- If "broken pipe" or signal deletion errors occur, check Worker signal lifecycle
- Ensure signals remain connected across multiple iterations
- Use `self.workers.append(worker)` to prevent garbage collection

### Branching Issues
- If duplicate messages appear, check conversation filtering logic in `ai_turn()`
- If images disappear on branch, this is a known GUI limitation - check `images/` folder
- Graph nodes overlapping: drag apart or disable physics with `apply_physics = False`

## Development Tips

- **Debugging conversation flow**: Add print statements in `ai_turn()` to trace message filtering
- **Testing new providers**: Start with simple single-turn conversations before branching
- **UI customization**: Colors defined in `COLORS` dict at top of gui.py
- **Async debugging**: Check Worker signals connect properly, use `finished.connect()` for cleanup
- **Branch logic**: Key distinction is in system prompt override for first 1-2 responses

## Code Quality

The project uses ruff for linting (configured in pyproject.toml):
```bash
# Check code
poetry run ruff check .

# Format code
poetry run ruff format .
```

