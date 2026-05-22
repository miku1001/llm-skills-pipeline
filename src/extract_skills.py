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
   hex_str = secrets.token_hex(8)
   return "-".join(hex_str[i:i+4] for i in range(0, 16, 4))

#Extract skill from single problem
def extract_skills(problem: str) -> list[dict]:
   prompt_template = load_prompt("extract_skills.txt")
   prompt = prompt_template.replace("{problem}", problem)

   response = client.chat.completions.create(
      model="openai/gpt-4o-mini",
      # model="nvidia/nemotron-3-super-120b-a12b:free",
      messages = [
         {'role': 'user', "content": prompt}
      ],
      temperature = 0.3
   )

   raw = response.choices[0].message.content

   #Parse JSON response
   parsed = json.loads(raw)
   listed_skill = parsed['skills']

   #attach uid 
   skills = []
   for skill in listed_skill:
      skills.append({
         "uid": generate_uid(),
         "skill_name": skill
      })

   return skills

#process multiple problem and save to skills.json
def run_extract_skill(problems: list[str]):
   all_skills = []
   seen_names = set()

   print(f"Extracting skills from {len(problems)} problems...\n")


   for i, problem in enumerate(problems):
    print(f"[{i+1}/{len(problems)}] Processing: {problem[:60]}...")
    try:
     skills = extract_skills(problem)

     for skill in skills:
          if skill["skill_name"] not in seen_names:
              seen_names.add(skill["skill_name"])
              all_skills.append(skill)
              print(f"  ✅ {skill['uid']} | {skill['skill_name']}")

    except Exception as e:
      print(f"  ❌ Error: {e}")
      continue

   # Save to data/skills.json
   output_path = Path("data") / "skills.json"
   with open(output_path, "w") as f:
        json.dump(all_skills, f, indent=2)

   print(f"\n Done! {len(all_skills)} unique skills saved to {output_path}")
   return all_skills