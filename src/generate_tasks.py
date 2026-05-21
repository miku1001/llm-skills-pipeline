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

#unique id
def generate_uid():
    return "-".join(secrets.token_hex(2) for _ in range(4))

#generate tasks each skill
def generate_tasks(skill: dict) -> list[dict]:
   prompt_template = load_prompt("generate_tasks.txt")
   prompt = prompt_template.replace("{skill_name}", skill['skill_name'])

   response = client.chat.completions.create(
      model="openai/gpt-4o-mini",
      # model="nvidia/nemotron-3-super-120b-a12b:free",
      messages = [
         {'role': 'user', "content": prompt}
      ],
      temperature = 0.7
   )

   raw = response.choices[0].message.content

   # Parse JSON response
   parsed = json.loads(raw)

    # Attach UIDs and link to skill
   tasks = []
   for task in parsed["tasks"]:
      tasks.append({
         "uid": generate_uid(),
         "skill_uid": skill["uid"],
         "skill_name": skill["skill_name"],
         "difficulty": task["difficulty"],
         "problem": task["problem"],
         "answer": task.get("answer", "")
      })

   return tasks
#process all skills.json
def run_generate_tasks():
     skills_path = Path('data') / "skills.json"
     with open(skills_path) as f:
        skills = json.load(f)

     print(f"Generating tasks for {len(skills)} skills...\n")

     all_tasks = []

     for i, skill in enumerate(skills):
        print(f"[{i+1}/{len(skills)}] Generating tasks for {skill['skill_name']} ")
        try:
           tasks = generate_tasks(skill)
           all_tasks.extend(tasks)
           for task in tasks:
              print(f"{task['difficulty'].upper()} : {task['problem'][:70]}")
        except Exception as e:
           print("ERROR!!!")
           continue
           

       # Outputs/tasks.json
     output_path = Path("data") / "tasks.json"
     with open(output_path, "w") as f:
          json.dump(all_tasks, f, indent=2)

     print(f"\n Done! {len(all_tasks)} unique skills saved to {output_path}")
     return all_tasks