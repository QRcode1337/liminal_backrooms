# liminal_backrooms

A Python-based application that enables dynamic, branching conversations between multiple AI models in a graphical user interface. Allows for forking and rabbitholing by selecting text and right clicking. The system supports various AI models including Claude, OpenAI, Gemini, Grok etc, allowing them to interact with each other through text and image generation.

Huge thanks to Andy Ayrey and Janus for their endless inspiration.

## Features

- Multi-model AI conversations with support for:
  - Claude (Anthropic)
  - OpenRouter Models:
    - GPT (OpenAI)
    - Grok (xAI)
    - LLaMA (Meta)
    - Gemini (Google)
    - Anything on openrouter - if it's not listed add in config.
  - OpenAI Images (gpt-image-1) for image generation (toggle in GUI)
  - OpenAI Sora 2 video generation (selectable as AI-2; videos saved to `videos/`)

- Dynamic Conversation Branching:
  - 🕳️ Rabbithole: Explore concepts in depth while retaining full context
  - 🔱 Fork: Continue conversations from specific points in new directions
  - Visual network graph showing conversation branches and connections
  - Drag-and-drop node organization
  - Automatic node spacing and collision avoidance
  - Easy navigation between branches
  - User can also interject at these points

- Advanced Features:
  - Chain of Thought reasoning display optional
  - Customizable conversation turns and modes (AI-AI or Human-AI)
  - Preset system prompt pairs
  - Image generation and analysis capabilities
  - Export functionality for conversations and generated images
  - Modern dark-themed GUI interface
  - Conversation memory system

## Prerequisites

- Python 3.10 or higher (but lower than 3.12)
- Poetry for dependency management
- Windows 10/11 or Linux (tested on Ubuntu 20.04+)

## API Keys Required

You'll need API keys from the following services to use all features:

1. Anthropic (Claude):
   - Sign up at: https://console.anthropic.com/
   - Endpoint: https://api.anthropic.com/v1/messages
   - Models: claude-3-opus, claude-3.5-sonnet, claude-3-haiku

2. OpenRouter:
   - Sign up at: https://openrouter.ai/
   - Endpoint: https://openrouter.ai/api/v1/chat/completions
   - Provides access to: GPT-4, Grok, Qwen, LLaMA, Gemini, and more

3. Replicate (for DeepSeek R1; optional):
   - Sign up at: https://replicate.com/
   - Used for DeepSeek R1 text generation; Flux image generation optional

4. OpenAI (Images and Sora video):
   - Requires `OPENAI_API_KEY`
   - Used for both Images (`gpt-image-1`) and Sora 2/Pro video generation
   - Optional: `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Create a `.env` file in the project root with your API keys (see Configuration section below)

## Configuration

1. Environment Variables (`.env`):
   - Create a `.env` file in the project root with your API keys:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   REPLICATE_API_TOKEN=your_replicate_token (not required)
   OPENAI_API_KEY=your_openai_api_key  # For image generation (gpt-image-1)
   ```

2. Application Configuration (`config.py`):
   - Runtime settings (e.g., turn delay)
   - Available AI models in `AI_MODELS` dictionary
   - System prompt pairs in `SYSTEM_PROMPT_PAIRS` dictionary
   - Add new models or prompt pairs by updating these dictionaries

3. Memory System (optional):
   - Place JSON files at `memories/ai-1_memories.json` and `memories/ai-2_memories.json`
   - Contents should be a JSON array of prior messages (simple strings are fine)

## Usage

1. Start the application:
```bash
poetry run python main.py
```

2. GUI Controls:
   - Mode Selection: Choose between AI-AI conversation or Human-AI interaction
   - Iterations: Set number of conversation turns (1-100)
   - AI Model Selection: Choose models for AI-1 and AI-2
   - Prompt Style: Select from predefined conversation styles
   - Input Field: Enter your message or initial prompt
   - Export: Save conversation and generated images

3. Branching Features:
   - Right-click on any text to access branching options:
     - 🕳️ Rabbithole: Explore a concept in depth
     - 🔱 Fork: Continue from a specific point
   - Click nodes in the network graph to navigate between branches
   - Adjust iterations and models on the fly without restarting the application
   - Drag nodes to organize your conversation map
   - Branches automatically space themselves for clarity
   - (Branching doesn't work very well with images in the GUI yet. The images disappear but will still be produced and can be found in the images folder.)

4. Special Features:
   - Chain of Thought: DeepSeek models show reasoning process
   - Image Generation: OpenAI Images (gpt-image-1) creates images from prompts
   - Export: Saves conversations and images with timestamps

### Using Sora 2 (Video Generation)

1. In `AI Model Selection`, set `AI-1` to an LLM and `AI-2` to `Sora 2` (or `Sora 2 Pro`).
2. In `Prompt Style`, choose `Video Collaboration (AI-1 to Sora)`.
   - `AI_2` prompt is intentionally blank (Sora does not use system prompts).
3. Start the session. On each AI-2 turn, Sora renders a video from the AI-1 prompt.
4. Output files are saved under `videos/` with timestamped filenames. The UI will print a line like:
   - `[Sora] Video created: videos/2025...mp4`
5. Note: Videos are not parsed back into context (yet); the next turn continues from text only.

Environment variables (optional):
```env
SORA_SECONDS=12        # clip duration (e.g., 4, 8, 10, 12)
SORA_SIZE=1280x720     # resolution hint (e.g., 1280x720)
OPENAI_BASE_URL=...    # override API base, if needed
```
For the auto-trigger mode (not required when using Sora as AI-2), you can also enable generating a Sora video after AI-1 responses:
```env
SORA_AUTO_FROM_AI1=1
```
This will run Sora in the background and save videos to `videos/` without using the GUI embedding.

## Troubleshooting

1. API Issues:
   - Check API key validity
   - Verify endpoint URLs in config
   - Check API rate limits
   - Monitor API response errors in console

2. GUI Issues:
   - Ensure PyQt6 is installed (handled by Poetry install)
   - Check Python version compatibility
   - Verify display resolution settings

3. Memory System:
   - Ensure memory files exist in `memories/`
   - Check JSON formatting
   - Monitor file permissions

4. Branching Issues:
   - If nodes overlap, try dragging them apart
   - If a branch seems stuck, try clicking propagate again
   - Check console for any error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Anthropic
- DeepSeek AI
- OpenRouter
- OpenAI
- Open-source contributors
- Andy Ayrey and Janus, both huge inspirations for this project

## Support

For issues and feature requests, please use the GitHub issue tracker.

