from datasets import load_dataset
from src.extract_skills import run as run_extract_skill
from src.generate_tasks import run as run_generate_tasks
from src.evaluate_answer import run as run_evaluate_answer
def load_gsm8k(num_problems=5):
    print("📥 Loading GSM8K dataset from HuggingFace...")
    
    # I-download ang dataset
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Kumuha ng unang N problems
    problems = [dataset[i]["question"] for i in range(num_problems)]
    
    print(f"✅ Loaded {len(problems)} problems from GSM8K\n")
    return problems

def main():
    print("\n" + "="*50)
    print("  LLM SKILLS PIPELINE — STARTING")
    print("="*50 + "\n")

    # Load real GSM8K problems
    problems = load_gsm8k(num_problems=3)

    # STEP 1
    print("📌 STEP 1: Extracting skills from problems...")
    print("-"*50)
    run_extract_skill(problems)

    # STEP 2
    print("\n📌 STEP 2: Generating tasks per skill...")
    print("-"*50)
    run_generate_tasks()

    # STEP 3
    print("\n📌 STEP 3: Running evaluations...")
    print("-"*50)
    run_evaluate_answer()

    # STEP 4
    print("\n📌 STEP 4: Analyzing results + generating heatmap...")
    print("-"*50)
    run_evaluate_answer()

    print("\n" + "="*50)
    print("  ✅ PIPELINE COMPLETE!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()