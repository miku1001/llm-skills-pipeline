import json
from pathlib import Path


def sort_skills_by_difficulty():
    # Load evaluations
    with open("data/evaluations.json") as f:
        evaluations = json.load(f)

    # Load skills
    with open("data/skills.json") as f:
        skills = json.load(f)

    # Calculate pass rate per skill_uid
    skill_stats = {}
    for eval in evaluations:
        uid = eval["skill_uid"]
        if uid not in skill_stats:
            skill_stats[uid] = {"passed": 0, "total": 0}
        skill_stats[uid]["total"] += 1
        if eval["is_correct"]:
            skill_stats[uid]["passed"] += 1

    # Attach pass rate to each skill
    for skill in skills:
        uid = skill["uid"]
        if uid in skill_stats:
            stats = skill_stats[uid]
            skill["pass_rate"] = round(stats["passed"] / stats["total"], 3)
            skill["passed"] = stats["passed"]
            skill["total"] = stats["total"]
        else:
            # No eval data yet for this skill
            skill["pass_rate"] = None
            skill["passed"] = 0
            skill["total"] = 0

    # Sort: highest pass rate first (easy), lowest last (hard)
    # Skills with no data go to end
    sorted_skills = sorted(
        skills,
        key=lambda s: s["pass_rate"] if s["pass_rate"] is not None else -1,
        reverse=True
    )

    # Save sorted skills
    output_path = Path("data") / "skills_sorted.json"
    with open(output_path, "w") as f:
        json.dump(sorted_skills, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  SKILL DIFFICULTY RANKING")
    print(f"{'='*50}")
    for i, skill in enumerate(sorted_skills):
        rate = skill["pass_rate"]
        bar = "🟢" if rate and rate >= 0.7 else "🟡" if rate and rate >= 0.4 else "🔴"
        rate_str = f"{rate*100:.0f}%" if rate is not None else "N/A"
        print(f"  {i+1:2d}. {bar} [{rate_str:>4}] {skill['skill_name']}")
    print(f"{'='*50}")
    print(f"\n💾 Saved to data/skills_sorted.json")

    return sorted_skills