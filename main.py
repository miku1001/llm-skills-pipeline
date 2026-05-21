from datasets import load_dataset
from src.extract_skills import run_extract_skill
from src.generate_tasks import run_generate_tasks
from src.evaluate_answer import run_evaluate_answer


#Pipeline
def load_gsm8k(num_problems=10):
    print("📥 Loading GSM8K dataset from HuggingFace...")
    
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Return list of dicts with question + answer to match evaluator expectations
    problems = [
        {"question": dataset[i]["question"], "answer": dataset[i].get("answer", "")} 
        for i in range(num_problems)
    ]

    print(f"✅ Loaded {len(problems)} problems from GSM8K\n")
    return problems

def main():
    print("\n" + "="*50)
    print("  LLM SKILLS PIPELINE — STARTING")
    print("="*50 + "\n")

    # Load real GSM8K problems
    problems = load_gsm8k()

    # STEP 1
    print("📌 STEP 1: Extracting skills from problems...")
    print("-"*50)
    run_extract_skill([p["question"] for p in problems])

    # STEP 2
    print("\n📌 STEP 2: Generating tasks per skill...")
    print("-"*50)
    run_generate_tasks()

    # STEP 3
    print("\n📌 STEP 3: Running evaluations...")
    print("-"*50)
    run_evaluate_answer(problems)

    print("\n" + "="*50)
    print("  ✅ PIPELINE COMPLETE!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()