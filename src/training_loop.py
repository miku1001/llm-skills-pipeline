import json
import time
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def load_prompt(filename):
    return (Path("prompts") / filename).read_text()

# ─────────────────────────────────────────
# STEP 1: Find failed tasks
# ─────────────────────────────────────────
def get_failed_tasks() -> list:
    with open("data/evaluations.json") as f:
        evaluations = json.load(f)
    
    failed = [e for e in evaluations if not e["is_correct"]]
    
    print(f"📋 Found {len(failed)} failed tasks:")
    for t in failed:
        print(f"  ❌ [{t['difficulty'].upper()}] {t['skill_name']}")
        print(f"     {t['problem'][:60]}...")
    
    return failed

# ─────────────────────────────────────────
# STEP 2: Generate one attempt
# ─────────────────────────────────────────
def generate_attempt(problem: str, correct_answer: str) -> dict:
    prompt = load_prompt("evaluate_answer.txt")\
        .replace("{problem}", problem)\
        .replace("{correct_answer}", correct_answer)

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    raw = response.choices[0].message.content.strip()
    
    # Clean markdown if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)

# ─────────────────────────────────────────
# STEP 3: Score an attempt
# ─────────────────────────────────────────
def score_attempt(attempt: dict) -> float:
    score = 0.0

    # Correct answer = biggest reward
    if attempt.get("is_correct"):
        score += 10.0

    # Reward detailed CoT
    steps = attempt.get("steps", [])
    score += len(steps) * 0.1

    # Reward complete skill labeling
    labeled = sum(1 for s in steps if s.get("skill_used"))
    if steps:
        score += (labeled / len(steps)) * 0.5

    return round(score, 3)

# ─────────────────────────────────────────
# STEP 4: Run training loop per task
# ─────────────────────────────────────────
def run_loop_for_task(task: dict, n_attempts: int = 16, keep_best: int = 4) -> dict:
    print(f"\n{'='*60}")
    print(f"🔄 TASK  : {task['skill_name']} [{task['difficulty'].upper()}]")
    print(f"   Problem: {task['problem'][:70]}...")
    print(f"   Before : ❌ FAIL (0% pass rate)")
    print(f"{'='*60}")

    attempts = []
    passed_count = 0

    for i in range(n_attempts):
        try:
            attempt = generate_attempt(task["problem"], task["correct_answer"])
            attempt["attempt_number"] = i + 1
            attempt["score"] = score_attempt(attempt)
            attempts.append(attempt)

            status = "✅" if attempt["is_correct"] else "❌"
            print(f"  Attempt {i+1:2d}: {status}  score={attempt['score']:.2f}  steps={len(attempt.get('steps', []))}")

            if attempt["is_correct"]:
                passed_count += 1

            time.sleep(0.3)  # avoid rate limiting

        except Exception as e:
            print(f"  Attempt {i+1:2d}: ⚠️  Error — {e}")
            continue

    # Sort by score, keep best 4
    best_attempts = sorted(attempts, key=lambda x: x["score"], reverse=True)[:keep_best]

    before_rate = 0.0
    after_rate = (passed_count / len(attempts) * 100) if attempts else 0.0

    print(f"\n  📊 RESULT:")
    print(f"  Before : 0%")
    print(f"  After  : {after_rate:.1f}% ({passed_count}/{len(attempts)} passed)")
    print(f"  ✅ Best {keep_best} attempts saved as training examples")

    return {
        "task_uid": task["task_uid"],
        "skill_name": task["skill_name"],
        "difficulty": task["difficulty"],
        "problem": task["problem"],
        "correct_answer": task["correct_answer"],
        "best_attempts": best_attempts,
        "before_pass_rate": before_rate,
        "after_pass_rate": after_rate,
        "total_attempts": len(attempts),
        "passed_attempts": passed_count
    }


def run_training_loop():
    print("\n" + "="*60)
    print("  TRAINING LOOP — STARTING")
    print("="*60 + "\n")

    # Step 1: Get failed tasks
    failed_tasks = get_failed_tasks()

    if not failed_tasks:
        print("\n✅ No failed tasks found — model is doing great!")
        return [], []

    # Step 2: Run loop for each failed task
    training_data = []
    improvement_report = []

    for task in failed_tasks:
        result = run_loop_for_task(task)
        training_data.append(result)
        improvement_report.append({
            "skill": result["skill_name"],
            "difficulty": result["difficulty"],
            "before": result["before_pass_rate"],
            "after": result["after_pass_rate"],
            "passed": result["passed_attempts"],
            "total": result["total_attempts"]
        })

    # Step 3: Save training data
    Path("data").mkdir(exist_ok=True)
    with open("data/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ TRAINING LOOP COMPLETE")
    print(f"   Tasks improved : {len(training_data)}")
    print(f"   Saved to       : data/training_data.json")
    print(f"{'='*60}\n")

    return training_data, improvement_report