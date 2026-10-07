#!/usr/bin/env python3
"""Quick import test to verify all dependencies are satisfied."""

import sys

def test_imports():
    """Test that all critical imports work."""
    print("Testing imports...")

    try:
        print("  - Importing config...")
        from config import (
            TURN_DELAY,
            AI_MODELS,
            SYSTEM_PROMPT_PAIRS,
            SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT,
            SHARE_CHAIN_OF_THOUGHT,
            DEVELOPER_TOOLS,
            OUTPUTS_DIR,
            SORA_SECONDS,
            SORA_SIZE,
            get_model_tier_by_id,
            get_model_id,
            get_invite_models_text
        )
        print("    [OK] config imports successful")

        print("  - Importing shared_utils...")
        from shared_utils import (
            call_claude_api,
            call_omniroute_api,
            call_openrouter_api,
            call_openai_api,
            generate_image_from_text
        )
        print("    [OK] shared_utils imports successful")

        print("  - Importing command_parser...")
        from command_parser import parse_commands
        print("    [OK] command_parser imports successful")

        print("  - Importing styles...")
        from styles import COLORS, FONTS, get_button_style
        print("    [OK] styles imports successful")

        print("  - Importing grouped_model_selector...")
        from grouped_model_selector import GroupedModelComboBox
        print("    [OK] grouped_model_selector imports successful")

        print("\n[SUCCESS] All imports successful!")
        print(f"   - DEVELOPER_TOOLS = {DEVELOPER_TOOLS}")
        print(f"   - OUTPUTS_DIR = {OUTPUTS_DIR}")
        print(f"   - AI_MODELS count = {len(AI_MODELS)}")
        print(f"   - SYSTEM_PROMPT_PAIRS count = {len(SYSTEM_PROMPT_PAIRS)}")

        return True

    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
