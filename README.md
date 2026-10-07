# liminal_backrooms

A Python-based application that enables dynamic conversations between multiple AI models in a graphical user interface. Originally designed for exploring liminal AI interactions, it's evolved into a flexible platform for multi-agent shenanigans.

## What's New

- **Cypher OS UI**: The app is restyled as a hacker-OS shell: ink-black background, acid-lime accent, thin grey borders, square window frames, faint scanlines. Each agent gets its own color (lime, ice, violet, rose, white) and the human operator is amber. Panes are renamed CTRL.PANEL, NET.GRAPH, SYS.MONITOR and MEDIA.VIEW. See [Fonts](#fonts) for the typefaces.
- **Per-Agent System Prompts**: A **SYS** button on each AI slot sets that agent's own system prompt, either added on top of the scenario prompt or replacing it
- **Briefing**: A shared prompt every agent reads once, before its first turn only
- **Group-Chat Tools in Every Scenario**: With **Agent tools + @mentions** on, every agent is told about the commands its scenario doesn't already list
- **`@AI-N` Mentions**: Agents can call on each other by slot. The mentioned agent gets an extra reply at the end of the round, or speaks next in step mode.
- **`!fetch "url"`**: Agents can read a public web page and share a text excerpt with the group
- **Non-Blocking Web Tools**: `!search` and `!fetch` run in the background, so the UI never freezes while they work. Search now uses the maintained `ddgs` package.
- **Conversation Search**: `Ctrl+F` opens a search overlay with match navigation and highlighting
- **Zoom**: `Ctrl+=` / `Ctrl+-` to zoom the chat pane (50%-200%), `Ctrl+0` to reset
- **Speed Controls**: 0.5x / 1x / 2x / 5x turn speed buttons in the status bar
- **Live Stats Panel**: New STATS tab showing turns, active AIs, estimated tokens, word count, avg response time, images generated, and commands executed
- **Control Panel**: `// SECTION` headers over thin dividers, and segmented slider gauges for iterations and number of AIs
- **Keyboard Shortcuts**: `Ctrl+Enter` (propagate), `Ctrl+E` (export), `Ctrl+Shift+N` (reset), `Escape` (stop), `F11` (fullscreen), `Ctrl+T` (toggle CRT)
- **Auto-Save Recovery**: Conversations auto-save every 30 seconds; on startup offers to recover the previous session
- **New AI Commands**:
  - `!vote "question" [option1, option2, ...]` — AIs can create polls
  - `!whisper "AI-2" "message"` — private to that AI (by slot, model id, or name). Other AIs don't see it; you do.
- **Message Tooltips**: Hover any message to see timestamp, model name, and estimated token count
- **Improved Scroll Stability**: Content fingerprinting and append-only rendering reduce scroll jitter during streaming
- **Dynamic AI Participants**: Models can invite other AIs into the conversation using `!add_ai` (up to 5 participants)
- **AI-Generated Images**: `!image` uses Grok Imagine (`grok-imagine-image-2.0`) and OpenAI GPT Image (`gpt-image-2`, Codex path), with OpenRouter Gemini as fallback
- **AI-Generated Videos**: Sora 2 video generation via `!video` command (currently disabled in scenarios — expensive!)
- **AI Self-Modification**: Models can modify their own system prompts (`!prompt`) and adjust their temperature (`!temperature`)
- **Web Search & Reading**: Models can search the internet (`!search`) and read pages (`!fetch`) for up-to-date information
- **BackroomsBench Evaluation (Beta)**: Multi-judge LLM evaluation system for measuring philosophical depth and linguistic creativity

## How It Works

All text LLMs run through **OmniRoute**. For Sora video generation, you'll need an **OpenAI API key**.

While great for AI shitposting, this is easy to customize for interesting experiments. Claude Opus 4.5 in Cursor (or similar) can whip up new scenarios in no time.

## Features

- Multi-model AI conversations with support for:

  - Claude (Anthropic) — all versions
  - GPT (OpenAI)
  - Grok (xAI)
  - Gemini (Google)
  - DeepSeek R1
  - Kimi K2
  - Anything exposed by OmniRoute — if it's not listed, add its exact live ID in config

- AI Agent Commands:

  - `!add_ai "Model Name" "persona"` — invite another AI to the conversation (max 5)
  - `!image "description"` — generate an image (Grok Imagine, or GPT Image when the speaker is Codex/OpenAI)
  - `!video "description"` — generate a video (Sora 2) [currently disabled in scenarios]
  - `!search "query"` — search the web for up-to-date information (runs in the background)
  - `!fetch "https://…"` — read a public web page and post a text excerpt to the chat (runs in the background; local and private-network addresses are blocked)
  - `@AI-N` — mention another agent by slot (e.g. `@AI-3`) to call on them. In normal mode they get an extra reply at the end of the round (max 2 per round, and those replies can't call on anyone else). In step mode they speak next. This also works when you type the mention yourself in step mode.
  - `!prompt "text"` — modify your own system prompt (persists across turns)
  - `!temperature X` — adjust your own sampling temperature (0-2, default 1.0)
  - `!mute_self` — sit out a turn and just listen
  - `!vote "question" [options]` — start a poll for the group
  - `!whisper "AI-X" "message"` — private message to a slot, model id, or name (`"grok 4.20"`). Visible to you, not to the other AIs.

- UI & Controls:
  - Cypher OS look with faint scanline overlay
  - Segmented sliders for iterations and number of AIs
  - Per-agent **SYS** prompt editor and a shared **BRIEFING** box
  - **Agent tools + @mentions** toggle (on by default)
  - Conversation search with `Ctrl+F`
  - Zoom with `Ctrl+=` / `Ctrl+-`
  - Speed controls (0.5x—5x) in status bar
  - Live stats tab (turns, tokens, response times, etc.)
  - Message tooltips with timestamp and token estimates
  - Auto-save with crash recovery

- Advanced Features:
  - Chain of Thought reasoning display (optional)
  - Customizable conversation turns and modes (AI-AI or Human-AI)
  - Preset scenario prompts for different vibes
  - Export functionality for conversations and generated images
  - Conversation memory system
  - BackroomsBench evaluation system (beta) with multi-judge LLM scoring

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Propagate (start/continue conversation) |
| `Ctrl+E` | Export conversation |
| `Ctrl+Shift+N` | Reset conversation |
| `Ctrl+F` | Search conversation |
| `Ctrl+=` / `Ctrl+-` | Zoom in / out |
| `Ctrl+0` | Reset zoom |
| `Ctrl+T` | Toggle CRT scanline effect |
| `Escape` | Stop current operation |
| `F11` | Toggle fullscreen |

## Prerequisites

- Python 3.10 or higher (but lower than 3.12)
- Poetry for dependency management
- Windows 10/11 or Linux (tested on Ubuntu 20.04+)

## API Keys Required

Create a `.env` file in the project root:

```env
OMNIROUTE_BASE_URL=http://127.0.0.1:20128/v1  # Required - all text LLMs route through here
OMNIROUTE_API_KEY=your_omniroute_api_key      # Optional for local OmniRoute unless auth is enabled
OPENROUTER_API_KEY=your_openrouter_api_key    # Optional - image generation fallback
OPENAI_API_KEY=your_openai_api_key            # Optional - only needed for Sora video generation
```

Get your keys:

- OpenRouter: https://openrouter.ai/
- OmniRoute local dashboard: http://127.0.0.1:20128/dashboard
- OpenAI (for Sora): https://platform.openai.com/

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd liminal_backrooms
```

2. Install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies using Poetry:

```bash
poetry install
```

4. Set up pre-commit hooks (for contributors):

```bash
poetry run pre-commit install
```

5. Create your `.env` file with API keys (see above)

## Usage

1. Start the application:

```bash
poetry run python main.py
```

2. GUI Controls:

   - Mode Selection: Choose between AI-AI conversation or Human-AI interaction
   - Iterations: Set number of conversation turns using the retro slider (click or scroll)
   - AI Model Selection: Choose models for each AI slot
   - Prompt Style: Select from predefined scenarios
   - SYS (on each AI row): Set that agent's own system prompt. Leave **Replace the scenario prompt** unchecked to add it on top of the scenario; check it to use only your prompt. The button reads `SYS ●` when a prompt is set.
   - Briefing: Shared text each agent receives once, before its first turn only. Use it for ground rules, the topic, or context everyone should start with.
   - Agent tools + @mentions (under Options): Gives every agent the group-chat tool list and turns on `@AI-N` mentions. Turn it off for a plain conversation.
   - Input Field: Enter your message or initial prompt
   - Export: Save conversation and generated images
   - View HTML: Open styled conversation in browser
   - BackroomsBench (beta): Run multi-judge evaluation on conversations

3. The AIs take it from there — they can add each other, generate images, and go wherever the scenario takes them.

## Configuration

Application settings in `config.py`:

- Runtime settings (turn delay, etc.)
- Available AI models in `AI_MODELS` dictionary
- Scenario prompts in `SYSTEM_PROMPT_PAIRS` dictionary

### Saved Prompts

Per-agent SYS prompts, the briefing text and the tools toggle are saved automatically with Qt's `QSettings` (under `LiminalBackrooms/CypherOS`) and restored on the next launch.

### Fonts

The UI prefers **JetBrains Mono** for text and **Chakra Petch** for headings. These fonts aren't bundled. To use them, drop the TTF files into `fonts/` with these names:

- `JetBrainsMono-Regular.ttf`, `JetBrainsMono-Bold.ttf`
- `ChakraPetch-SemiBold.ttf`, `ChakraPetch-Bold.ttf`

Without them the app falls back to the bundled Iosevka Term.

### Developer Tools

For debugging the GUI, set `DEVELOPER_TOOLS = True` in `config.py`. This enables:

- **F12**: Toggle debug inspector panel
- **Ctrl+Shift+C**: Pick and inspect any UI element

Keep this `False` for normal usage.

### Adding New Models

Add entries to `_CURATED_MODELS` in `config.py` using **exact live OmniRoute IDs** from `omniroute models` or `GET $OMNIROUTE_BASE_URL/models`. OpenRouter-style IDs (for example `anthropic/claude-opus-4.6`) will 404.

```python
"Claude Opus 4.6": "anthropic/claude-opus-4-6",
"GPT 5.6 Sol (High)": "cx/gpt-5.6-sol-high",
```

The model dropdown also has an **Endpoints** group built from live OmniRoute prefixes. Choose the route, not just the model:

- `agy/…` Antigravity
- `cx/…` or `codex/…` Codex
- `xai/…` xAI Grok
- `grok-cli/…` Grok CLI
- `anthropic/…`, `gemini/…`, `openai/…`, `gh/…`, `nvidia/…`, `auto/…`

### Creating Custom Scenarios

Add entries to `SYSTEM_PROMPT_PAIRS` in config.py. Each scenario needs prompts for AI-1 through AI-5. Check existing scenarios for the format — or just ask an AI to write them for you.

## Sora 2 Video Generation

To enable video generation:

1. Set one AI slot to `Sora 2` or `Sora 2 Pro`
2. Or add `!video` commands to your scenario prompts
3. Videos save to `videos/` folder

Environment variables (optional):

```env
SORA_SECONDS=12        # clip duration (4, 8, 10, 12)
SORA_SIZE=1280x720     # resolution
```

**Note**: Video generation is expensive. The `!video` command has been removed from default scenarios but is easy to add back.

## Troubleshooting

1. API Issues:

   - Confirm OmniRoute is healthy: `omniroute --output json health`
   - Check `OMNIROUTE_BASE_URL` in `.env` (default `http://127.0.0.1:20128/v1`)
   - Use a live OmniRoute model ID, not an OpenRouter ID
   - Image generation still uses `OPENROUTER_API_KEY` when configured

2. Web Tool Issues:

   - `!search` uses the `ddgs` package. If search returns nothing, check that `ddgs` is installed (`poetry install`). The macOS `.app` launcher uses `.runtime-venv`, so install it there too: `uv pip install --python .runtime-venv/bin/python ddgs`
   - `!fetch` only reads public `http(s)` pages. It refuses localhost and private-network addresses (also after redirects), and non-text content.

3. GUI Issues:

   - Ensure PyQt6 is installed (handled by Poetry install)
   - Check Python version compatibility

4. Empty Responses:
   - Some models occasionally return empty — the app will retry once automatically
   - Check OmniRoute logs / dashboard if persistent

## Contributing

1. Fork the repository
2. Clone and install dependencies:
   ```bash
   poetry install
   poetry run pre-commit install
   ```
3. Create a feature branch
4. Make your changes
5. Commit (pre-commit hooks will run automatically)
6. Push and create a Pull Request

**Note**: The pre-commit hook will block commits if `DEVELOPER_TOOLS = True` in config.py. Make sure to set it back to `False` before committing.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
