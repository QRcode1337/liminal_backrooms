Original prompt: okay so I want you to spin up liminal backrooms in its entirety

## Startup log

- Project identified as PyQt6 desktop application launched by `poetry run python main.py`.
- Initial startup blocked because Poetry launcher references removed Homebrew Python 3.14.2.
- Existing `.venv` also stalled while reading packages, so a clean Python 3.11 runtime was created at `.runtime-venv` with project dependencies.
- Required provider configuration is present in `.env`; values were not printed.
- `test_imports.py` passed all critical imports. Configuration loaded Synthetic Consciousness and refreshed OpenRouter model cache (32 curated models kept, 53 stale IDs removed).
- Application launched with `.runtime-venv/bin/python -u main.py` and remains healthy as PID 13851.
- macOS window verified: `╔═ LIMINAL BACKROOMS v0.7 ═╗`; window brought to front.
- No crash or freeze logs were produced.
- Non-blocking warning: bundled Iosevka fonts are absent, so Qt uses system fonts.

## OmniRoute integration

- Text chat and BackroomsBench evaluation requests now use the local OmniRoute OpenAI-compatible endpoint at `http://127.0.0.1:20128/v1`; Sora and media keep their specialized paths.
- The selector now puts an `Endpoints` tier first, built from the cached/live OmniRoute catalog. It includes the requested `cx/gpt-5.6-*` and `agy/gemini-3.5-flash-*` routes, along with other configured provider prefixes.
- The verified AGY Flash family is Gemini 3.5; there is no `agy/gemini-3.6-flash` route.
- Added OmniRoute settings/health check, live-catalog validation, model routing tests, and corrected branch selections to send model IDs instead of UI labels.
- Offline checks passed: Python compilation and 10 OmniRoute tests. Current sandbox blocks loopback requests, so this session cannot repeat the final live request/GUI restart without a host-level run.

## TODO

- Optional maintenance: repair global Poetry launcher, which points at removed Homebrew Python 3.14.2.
- Optional visual polish: restore bundled Iosevka font files.
- Restart the already-running PyQt app from a host terminal to load OmniRoute changes, then select an `Endpoints` model and send a short message.
