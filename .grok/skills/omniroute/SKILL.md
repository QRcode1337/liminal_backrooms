---
name: omniroute
description: >
  Route Liminal Backrooms text LLMs through local OmniRoute using exact live
  model IDs. Use when adding models, fixing chat API calls, changing provider
  routing, editing iOS chat, or the user mentions OmniRoute, OMNIROUTE_BASE_URL,
  or model catalogs.
---

# OmniRoute in Liminal Backrooms

All **text** chat goes through local OmniRoute. OmniRoute owns provider credentials, fallback, and health. Do not add new direct Anthropic/OpenAI/xAI chat paths.

## Endpoints

- Chat: `POST $OMNIROUTE_BASE_URL/chat/completions`
- Catalog: `GET $OMNIROUTE_BASE_URL/models`
- Default base URL: `http://127.0.0.1:20128/v1`
- Optional auth: `OMNIROUTE_API_KEY`

Use `shared_utils.call_omniroute_api` on desktop. iOS uses `OmniRouteClient`.

## Model IDs

IDs must match live OmniRoute output (`omniroute models` or GET `/models`). Examples that currently work:

- `anthropic/claude-opus-4-6` (hyphens, not `anthropic/claude-opus-4.6`)
- `gemini/gemini-3.1-pro-preview` (not `google/gemini-3-pro-preview`)
- `xai/grok-4.6` (not `x-ai/grok-4`)
- `cx/gpt-5.6-sol-high`, `agy/gemini-3.5-flash-medium`

The desktop picker has an **Endpoints** tier built from live OmniRoute prefixes: `agy`, `cx`, `codex`, `xai`, `grok-cli`, `anthropic`, `gemini`, `openai`, `cerebras`, `gh`, `nvidia`, `auto`. Choosing `agy/gemini-3.5-flash-medium` routes through Antigravity; `cx/gpt-5.6-sol-high` through Codex; `xai/grok-4.6` through xAI.

Add curated shortcuts in `config.py` `_CURATED_MODELS`. `tools/model_updater.py` drops IDs OmniRoute no longer exposes and rebuilds the Endpoints group. Keep `sora-2` / `sora-2-pro` as OpenAI media IDs.

Do not use `auto` or Fusion combos as the default for independent multi-agent slots unless the user asked for a combo.

## Out of OmniRoute

- Sora video: `OPENAI_API_KEY`
- Image generation fallback: `OPENROUTER_API_KEY`

## Checks

```sh
omniroute --output json health
python -m unittest test_omniroute.py
```
