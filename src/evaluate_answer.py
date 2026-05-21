import json
import os
import secrets
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

#get prompt template
def load_prompt(filename):
  prompt_path = Path('prompts') / filename
  return prompt_path.read_text()

# Generate a simple correct answer using LLM
# Testing purpose
def get_correct_answer(problem: str) -> str:
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[
            {
                "role": "user",
                "content": f"Solve this math problem. Reply with the final numeric answer ONLY, nothing else.\n\nProblem: {problem}"
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

#evaluate answer 
def evaluate_answer(task: dict) -> dict:
   prompt_template = load_prompt("evaluate_answer.txt")
   correct_answer = get_correct_answer(task["problem"])
   prompt = prompt_template.replace("{problem}", task['problem']).replace("{correct_answer}", correct_answer)

   response = client.chat.completions.create(
      model="openai/gpt-4o-mini",
      # model="nvidia/nemotron-3-super-120b-a12b:free",
      messages = [
         {'role': 'user', "content": prompt}
      ],
      temperature = 0.3
   )

   raw = response.choices[0].message.content.strip()

    # If model wrapped output in Markdown code fences, extract inner content
   if raw.startswith("```"):
        try:
            inner = raw.split("```", 2)[1]
        except IndexError:
            inner = raw
        raw = inner
        if raw.lstrip().lower().startswith("json"):
             raw = raw.lstrip()[4:]

    # Parse JSON response
   parsed = json.loads(raw.strip())

    # Combine task info + evaluation result
   return {
      "task_uid": task["uid"],
      "skill_uid": task["skill_uid"],
      "skill_name": task["skill_name"],
      "difficulty": task["difficulty"],
      "problem": task["problem"],
      "correct_answer": correct_answer,
      "steps": parsed["steps"],
      "final_answer": parsed["final_answer"],
      "is_correct": parsed["is_correct"],
      "num_steps": len(parsed["steps"])
      }

#process all tasks
def run_evaluate_answer():
     tasks_path = Path('data') / "tasks.json"
     with open(tasks_path) as f:
        tasks = json.load(f)

     print(f"Evaluating {len(tasks)} tasks...\n")

     all_evaluations = []
     passed = 0
     failed = 0

     for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] [{task['difficulty'].upper()}] {task['problem'][:60]}...")

        try:
           result = evaluate_answer(task)
           all_evaluations.append(result)

           status = "✅ PASS" if result["is_correct"] else "❌ FAIL"
           print(f"  {status} | Steps: {result['num_steps']} | Answer: {result['final_answer']}")

           if result['is_correct']:
               passed += 1
           else:
               failed += 1
        except Exception as e:
           print(f"ERROR: {e}")
           continue
           

     # Outputs/evaluations.json
     output_path = Path("data") / "evaluations.json"
     with open(output_path, "w") as f:
        json.dump(all_evaluations, f, indent=2)

     print(f"\n Done! {len(all_evaluations)} evaluations saved to {output_path} ({passed} passed, {failed} failed)")
     return all_evaluations