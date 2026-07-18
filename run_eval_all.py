"""
Batch-evaluate every response file under responses/<model>/*.jsonl with the
local Ollama critic model, mirroring outputs to score_dir/<model>/*.jsonl.

Usage:
    python run_eval_all.py
    python run_eval_all.py --evaluator critic --responses_dir responses --score_dir score_dir
"""
import os
import glob
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Batch-run WritingBench evaluation over all response files.")
    parser.add_argument("--evaluator", choices=["critic", "claude"], default="critic")
    parser.add_argument("--query_criteria_file", default="benchmark_query/benchmark_all.jsonl")
    parser.add_argument("--responses_dir", default="responses")
    parser.add_argument("--score_dir", default="score_dir")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.responses_dir, "**", "*.jsonl"), recursive=True))
    if not files:
        print(f"No response files found under {args.responses_dir}/")
        return

    print(f"Found {len(files)} response file(s):")
    for f in files:
        print("  ", f)

    for input_file in files:
        rel = os.path.relpath(input_file, args.responses_dir)
        out_file = os.path.join(args.score_dir, rel)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        print(f"\n=== Evaluating {input_file} -> {out_file} ===")
        cmd = [
            sys.executable, "evaluate_benchmark.py",
            "--evaluator", args.evaluator,
            "--query_criteria_file", args.query_criteria_file,
            "--input_file", input_file,
            "--output_file", out_file,
        ]
        subprocess.run(cmd, check=True)

    print("\nAll done.")

if __name__ == "__main__":
    main()
