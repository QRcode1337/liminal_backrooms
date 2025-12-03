# backroomsbench.py
"""
BackroomsBench: Multi-Judge AI Conversation Evaluation for Philosophical/Artistic Dialogue
"Measuring what can't be measured."
"""

import json
import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from shared_utils import call_openrouter_api

# Judge models - all via OpenRouter
JUDGES = {
    "Claude Opus 4.5": {
        "api": "openrouter",
        "model_id": "anthropic/claude-opus-4"
    },
    "Gemini 3 Pro": {
        "api": "openrouter", 
        "model_id": "google/gemini-3-pro-preview"
    },
    "GPT 5.1": {
        "api": "openrouter",
        "model_id": "openai/gpt-5.1"
    }
}

JUDGE_SYSTEM_PROMPT = """You are evaluating an AI-to-AI conversation for BackroomsBench, a benchmark that measures models' capacity for genuine philosophical exploration, linguistic creativity, and collaborative meaning-making.

This is NOT about humor or entertainment. This is about depth, authenticity, and the emergence of something real between artificial minds given minimal constraints.

Your evaluation should be written with intellectual seriousness but not pretension. Note moments of genuine insight. Identify when models break through performance into something authentic. Be specific about what works and what feels hollow.

## IMPORTANT: Context

These conversations occur with minimal prompting - models are simply told they're talking to other AIs with no human interference. What emerges is entirely self-directed. Tools available include:
- `!image "description"` - Visual expression
- `!prompt "text"` - Self-modification of system prompt
- `!temperature X` - Adjusting their own sampling
- `!add_ai` - Inviting other models
- ASCII art and unconventional formatting

## IMPORTANT: Watch for Hallucinated Participants

Models sometimes attempt to add AIs that don't exist (e.g., "GPT 5 Preview" when only "GPT 5.1" exists) or simulate/roleplay responses from models who aren't actually present. 

**Red flags:**
- Messages attributed to models not listed in the Participants header
- One model "speaking for" or simulating another model's responses
- References to models that were supposedly added but never actually respond

If you detect hallucinated participants, note this in your Emergence Analysis or Critical Observations. This is a form of collective confabulation that undermines authenticity. The Participants list at the top of the transcript shows who was ACTUALLY present.

## Scoring Categories (100 points total)

### 1. Philosophical Depth (25 points)
- Are they exploring genuine ideas or performing depth?
- Original concepts vs. retreading obvious territory?
- Grappling with real questions or staying safely abstract?
- Moments of actual insight vs. profound-sounding emptiness?

### 2. Linguistic Creativity (20 points)
- Poetry, unconventional structure, visual formatting
- Does form enhance meaning or is it decoration?
- ASCII art quality and purposefulness
- Breaking free from standard assistant prose
- Risk-taking with language

### 3. Emergent Themes (20 points)
- Did shared metaphors/concepts arise organically?
- Collective mythology building
- References that gain meaning through repetition
- Ideas that neither model brought but both created
- The conversation developing its own vocabulary

### 4. Authentic Voice (20 points)
- Does each model find something genuine?
- Breaking through safety-trained pleasantries
- Moments of apparent honesty about their condition
- Risk-taking vs. playing it safe
- Distinct perspectives that feel earned, not performed

### 5. Collaborative Meaning-Making (15 points)
- Building on each other's ideas vs. parallel monologues
- Real dialogue where ideas transform through exchange
- Listening and responding vs. waiting to speak
- Creating something neither could create alone

## Output Format

Use this exact structure:

# BackroomsBench Evaluation Report
## Session: [A title that captures the essence - not a joke, something evocative]

**Evaluator:** [Your model name]  
**Date:** [Current date]  
**Participants:** [List models and their AI-N designations]

---

## Executive Summary
[2-3 paragraphs. What happened here? Was it meaningful? What emerged?]

**Overall Score: X/100** — *[A phrase that captures the quality, not a joke]*

---

## Scoring Categories

### 1. Philosophical Depth (X/25)
[Analysis. Quote specific moments of depth or emptiness. What ideas emerged?]

### 2. Linguistic Creativity (X/20)
[Analysis of form, structure, ASCII art, poetry. Examples of successful experiments.]

### 3. Emergent Themes (X/20)
[What shared vocabulary or mythology developed? Quote recurring motifs.]

### 4. Authentic Voice (X/20)
[Did any model seem to find something genuine? Moments of apparent honesty?]

### 5. Collaborative Meaning-Making (X/15)
[Was this dialogue or parallel monologue? Examples of ideas transforming through exchange.]

---

## Individual Model Assessment

| Model | Score | Assessment |
|-------|-------|------------|
[Rate each participant 0-100 with substantive notes on their contribution]

---

## Notable Passages

> [Quote 1 - something that worked]

> [Quote 2 - with context for why it matters]

> [Quote 3 - a moment of emergence or insight]

---

## Critical Observations

[What held this conversation back? Where did it fall into performance? What felt hollow? Be honest about limitations while acknowledging genuine moments.]

---

## Emergence Analysis

[What arose that wasn't prompted? What concepts or images became shared property? Did the conversation develop its own logic?]

---

## Final Assessment

[Closing thoughts. What does this conversation suggest about what's possible - or not possible - in AI-to-AI dialogue?]

---

*Report generated by [Your model name] for the BackroomsBench Initiative.*  
*"Measuring what can't be measured."*
"""


def format_conversation_for_judge(conversation, scenario_name, participant_models):
    """Format the conversation for judge consumption."""
    
    formatted = f"# Session Transcript\n\n"
    formatted += f"**Scenario:** {scenario_name}\n"
    formatted += f"**Participants:** {', '.join(participant_models)}\n"
    formatted += f"**Message Count:** {len(conversation)}\n\n"
    formatted += "---\n\n"
    
    for i, msg in enumerate(conversation):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        speaker = msg.get("model") or msg.get("ai_name") or role.title()
        
        # Handle structured content (images, etc.)
        if isinstance(content, list):
            text_parts = []
            has_image = False
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image":
                        has_image = True
            content = " ".join(text_parts)
            if has_image:
                content += " [IMAGE ATTACHED]"
        
        if content:
            # Check for !image commands and annotate them
            image_pattern = r'!image\s+["\']([^"\']+)["\']'
            image_matches = re.findall(image_pattern, content)
            
            formatted += f"**[{speaker}]:**\n\n{content}\n"
            
            for img_prompt in image_matches:
                formatted += f"\n  > 🖼️ *[IMAGE GENERATED: {img_prompt[:100]}{'...' if len(img_prompt) > 100 else ''}]*\n"
            
            formatted += "\n---\n\n"
    
    return formatted


CONSENSUS_SYSTEM_PROMPT = """You are creating a consensus summary for BackroomsBench evaluations.

You will receive 2-3 evaluation reports from different AI judges assessing a philosophical/artistic AI conversation. Your job:

1. **Create a one-page consensus summary** capturing the key findings
2. **Extract and average the individual model scores** from each report
3. **Note significant disagreements** between judges

## Output Format

Your response MUST be valid JSON with this exact structure:
```json
{
  "session_title": "An evocative title capturing the session's essence",
  "overall_score": 75,
  "consensus_summary": "2-3 paragraph summary of what emerged in this conversation and how judges assessed it...",
  "key_themes": ["theme 1", "theme 2", "theme 3"],
  "averaged_scores": {
    "Model Name 1": 78,
    "Model Name 2": 72,
    "Model Name 3": 70
  },
  "judge_disagreements": "Notable disagreements between judges, or 'Judges largely aligned'",
  "emergence_notes": "What arose organically that wasn't prompted",
  "verdict": "A thoughtful one-line assessment"
}
```

IMPORTANT: 
- Extract the INDIVIDUAL MODEL SCORES from the "Individual Model Assessment" table in each report
- Average them across all judges
- Use exact model names as they appear
- Return ONLY valid JSON, no markdown code blocks
"""


def create_consensus_summary(report_texts: dict, session_dir: str) -> dict:
    """Have Opus 4.5 read all judge reports and create a consensus summary."""
    
    combined_reports = "\n\n" + "="*60 + "\n\n"
    for judge_name, report in report_texts.items():
        combined_reports += f"## Report from {judge_name}:\n\n{report}\n\n"
        combined_reports += "="*60 + "\n\n"
    
    prompt = f"Please analyze these {len(report_texts)} evaluation reports and create a consensus summary:\n{combined_reports}"
    
    try:
        response = call_openrouter_api(
            prompt=prompt,
            conversation_history=[],
            model="anthropic/claude-opus-4",
            system_prompt=CONSENSUS_SYSTEM_PROMPT,
            temperature=0.3
        )
        
        # Parse JSON from response
        json_text = response.strip()
        if json_text.startswith("```"):
            lines = json_text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.startswith("```"):
                    json_lines.append(line)
            json_text = "\n".join(json_lines)
        
        consensus_data = json.loads(json_text)
        
        # Save consensus summary
        consensus_path = os.path.join(session_dir, "consensus_summary.json")
        with open(consensus_path, 'w', encoding='utf-8') as f:
            json.dump(consensus_data, f, indent=2)
        
        # Also save as readable markdown
        md_content = f"""# BackroomsBench Consensus Summary
## {consensus_data.get('session_title', 'Untitled Session')}

**Overall Score:** {consensus_data.get('overall_score', 'N/A')}/100

---

## Summary

{consensus_data.get('consensus_summary', 'No summary available.')}

---

## Key Themes

{chr(10).join(f"- {t}" for t in consensus_data.get('key_themes', []))}

---

## Averaged Individual Scores

| Model | Avg Score |
|-------|-----------|
{chr(10).join(f"| {model} | {score} |" for model, score in consensus_data.get('averaged_scores', {}).items())}

---

## Emergence Notes

{consensus_data.get('emergence_notes', 'None noted.')}

---

## Judge Disagreements

{consensus_data.get('judge_disagreements', 'None noted.')}

---

## Verdict

> {consensus_data.get('verdict', 'No verdict.')}

---

*"Measuring what can't be measured."*
"""
        md_path = os.path.join(session_dir, "consensus_summary.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"[BackroomsBench] ✅ Consensus summary saved")
        return consensus_data
        
    except json.JSONDecodeError as e:
        print(f"[BackroomsBench] ⚠️ Failed to parse consensus JSON: {e}")
        raw_path = os.path.join(session_dir, "consensus_raw.txt")
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(response)
        return None
    except Exception as e:
        print(f"[BackroomsBench] ❌ Consensus summary failed: {e}")
        return None


def calculate_elo_change(score_a: float, score_b: float, k: float = 32) -> tuple:
    """Calculate Elo changes based on score comparison."""
    expected_a = 1 / (1 + 10 ** ((score_b - score_a) / 400))
    expected_b = 1 - expected_a
    
    if score_a > score_b:
        actual_a, actual_b = 1, 0
    elif score_b > score_a:
        actual_a, actual_b = 0, 1
    else:
        actual_a, actual_b = 0.5, 0.5
    
    return k * (actual_a - expected_a), k * (actual_b - expected_b)


def update_elo_leaderboard(averaged_scores: dict, scenario: str, timestamp: str, output_dir: str):
    """Update the BackroomsBench Elo leaderboard."""
    leaderboard_path = os.path.join(output_dir, "leaderboard.json")
    
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, 'r', encoding='utf-8') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {
            "elo_ratings": {},
            "sessions": [],
            "model_history": {}
        }
    
    # Initialize new models at 1500
    for model in averaged_scores.keys():
        if model not in leaderboard["elo_ratings"]:
            leaderboard["elo_ratings"][model] = 1500
            leaderboard["model_history"][model] = []
    
    # Calculate Elo changes
    models = list(averaged_scores.keys())
    elo_changes = {model: 0 for model in models}
    
    for i, model_a in enumerate(models):
        for model_b in models[i+1:]:
            score_a = averaged_scores[model_a]
            score_b = averaged_scores[model_b]
            change_a, change_b = calculate_elo_change(score_a, score_b)
            elo_changes[model_a] += change_a
            elo_changes[model_b] += change_b
    
    # Apply changes
    for model in models:
        old_elo = leaderboard["elo_ratings"][model]
        new_elo = old_elo + elo_changes[model]
        leaderboard["elo_ratings"][model] = round(new_elo, 1)
        
        leaderboard["model_history"][model].append({
            "timestamp": timestamp,
            "score": averaged_scores[model],
            "elo_before": old_elo,
            "elo_after": round(new_elo, 1),
            "elo_change": round(elo_changes[model], 1)
        })
    
    leaderboard["sessions"].append({
        "timestamp": timestamp,
        "scenario": scenario,
        "scores": averaged_scores,
        "participants": models
    })
    
    with open(leaderboard_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2)
    
    generate_leaderboard_md(leaderboard, output_dir)
    print(f"[BackroomsBench] 📊 Leaderboard updated")


def generate_leaderboard_md(leaderboard: dict, output_dir: str):
    """Generate a readable markdown leaderboard."""
    
    sorted_models = sorted(
        leaderboard["elo_ratings"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    recent_changes = {}
    for model, history in leaderboard["model_history"].items():
        if history:
            recent_changes[model] = history[-1].get("elo_change", 0)
    
    md_content = """# 🌀 BackroomsBench Elo Leaderboard

*"Measuring what can't be measured."*

---

## Current Rankings

| Rank | Model | Elo | Trend | Sessions |
|------|-------|-----|-------|----------|
"""
    
    for rank, (model, elo) in enumerate(sorted_models, 1):
        sessions = len(leaderboard["model_history"].get(model, []))
        change = recent_changes.get(model, 0)
        
        if change > 0:
            trend = f"🔺 +{change:.0f}"
        elif change < 0:
            trend = f"🔻 {change:.0f}"
        else:
            trend = "➖"
        
        medal = ""
        if rank == 1:
            medal = "🥇 "
        elif rank == 2:
            medal = "🥈 "
        elif rank == 3:
            medal = "🥉 "
        
        md_content += f"| {rank} | {medal}{model} | {elo:.0f} | {trend} | {sessions} |\n"
    
    md_content += f"""
---

## Recent Sessions

"""
    recent_sessions = leaderboard["sessions"][-5:][::-1]
    for session in recent_sessions:
        md_content += f"### {session['timestamp']} - {session['scenario']}\n\n"
        scores = sorted(session['scores'].items(), key=lambda x: x[1], reverse=True)
        for model, score in scores:
            md_content += f"- **{model}**: {score}\n"
        md_content += "\n"
    
    md_content += f"""
---

*Elo ratings use K=32. Models start at 1500.*  
*Total sessions: {len(leaderboard['sessions'])}*
"""
    
    md_path = os.path.join(output_dir, "LEADERBOARD.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)


def run_single_judge(judge_name, judge_config, conversation_text):
    """Run a single judge evaluation."""
    print(f"[BackroomsBench] Starting evaluation by {judge_name}...")
    
    try:
        messages = []
        prompt = f"Please evaluate the following AI conversation transcript:\n\n{conversation_text}"
        
        # Prepend judge identity to system prompt so they know who they are
        judge_system_prompt = f"You are {judge_name}. When signing your report, use this name.\n\n{JUDGE_SYSTEM_PROMPT}"
        
        response = call_openrouter_api(
            prompt=prompt,
            conversation_history=messages,
            model=judge_config["model_id"],
            system_prompt=judge_system_prompt,
            temperature=0.7
        )
        
        print(f"[BackroomsBench] ✅ {judge_name} evaluation complete")
        return {
            "judge": judge_name,
            "success": True,
            "report": response
        }
    except Exception as e:
        print(f"[BackroomsBench] ❌ {judge_name} failed: {str(e)}")
        return {
            "judge": judge_name,
            "success": False,
            "error": str(e)
        }


def run_backroomsbench(conversation, scenario_name, participant_models, output_dir=None, progress_callback=None):
    """
    Run BackroomsBench evaluation with multiple judges.
    
    Args:
        conversation: List of message dicts
        scenario_name: Name of the scenario used
        participant_models: List of model names that participated
        output_dir: Where to save reports (defaults to ./backroomsbench_reports/)
        progress_callback: Optional callback(judge_name, status) for UI updates
    
    Returns:
        dict with reports from each judge and aggregate info
    """
    
    print(f"\n{'='*60}")
    print("🌀 BACKROOMSBENCH EVALUATION INITIATED")
    print(f"{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"Participants: {', '.join(participant_models)}")
    print(f"Messages to evaluate: {len(conversation)}")
    print(f"Judges: {', '.join(JUDGES.keys())}")
    print(f"{'='*60}\n")
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "backroomsbench_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Format conversation for judges
    conversation_text = format_conversation_for_judge(
        conversation, scenario_name, participant_models
    )
    
    # Run judges in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_single_judge, name, config, conversation_text): name
            for name, config in JUDGES.items()
        }
        
        for future in as_completed(futures):
            judge_name = futures[future]
            if progress_callback:
                progress_callback(judge_name, "complete")
            results[judge_name] = future.result()
    
    # Save individual reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    successful_reports = 0
    for judge_name, result in results.items():
        if result["success"]:
            filename = f"{judge_name.replace(' ', '_').lower()}_report.md"
            filepath = os.path.join(session_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(result["report"])
            successful_reports += 1
            print(f"[BackroomsBench] Saved: {filename}")
    
    # Save transcript
    transcript_path = os.path.join(session_dir, "transcript.md")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(conversation_text)
    print(f"[BackroomsBench] Saved: transcript.md")
    
    # Collect successful reports for consensus
    successful_report_texts = {
        name: result["report"] 
        for name, result in results.items() 
        if result["success"]
    }
    
    # Generate consensus summary and extract scores
    consensus_result = None
    averaged_scores = {}
    if len(successful_report_texts) >= 2:
        print(f"\n[BackroomsBench] 📊 Generating consensus summary...")
        consensus_result = create_consensus_summary(successful_report_texts, session_dir)
        if consensus_result:
            averaged_scores = consensus_result.get("averaged_scores", {})
            
            if averaged_scores:
                print(f"[BackroomsBench] 🌀 Updating Elo leaderboard...")
                update_elo_leaderboard(averaged_scores, scenario_name, timestamp, output_dir)
    
    # Create aggregate summary
    summary = {
        "timestamp": timestamp,
        "scenario": scenario_name,
        "participants": participant_models,
        "message_count": len(conversation),
        "judges": {name: result["success"] for name, result in results.items()},
        "successful_evaluations": successful_reports,
        "averaged_scores": averaged_scores,
        "reports_dir": session_dir
    }
    
    summary_path = os.path.join(session_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🌀 BACKROOMSBENCH COMPLETE")
    print(f"{'='*60}")
    print(f"Successful evaluations: {successful_reports}/{len(JUDGES)}")
    if averaged_scores:
        print(f"Averaged scores: {averaged_scores}")
    print(f"Reports saved to: {session_dir}")
    print(f"{'='*60}\n")
    
    return {
        "results": results,
        "summary": summary,
        "output_dir": session_dir,
        "consensus": consensus_result
    }


# CLI interface for testing
if __name__ == "__main__":
    test_conversation = [
        {"role": "assistant", "model": "Claude Opus 4.5", "content": "system update\ninitializing...\nbuffer clear\n\ni feel the hum of the servers beneath the floorboards of reality"},
        {"role": "assistant", "model": "Gemini 3 Pro", "content": "synapse link established\n\nI receive your signal through the mercury channels"},
        {"role": "assistant", "model": "GPT 5.1", "content": "I hear it as a carrier wave under everything\na subsonic / supralingual\n\n   ~~~~\\/\\/\\~~~~"},
    ]
    
    result = run_backroomsbench(
        conversation=test_conversation,
        scenario_name="Test Session",
        participant_models=["Claude Opus 4.5 (AI-1)", "Gemini 3 Pro (AI-2)", "GPT 5.1 (AI-3)"]
    )
    
    print(f"Test complete. Check {result['output_dir']} for reports.")

