"""
Standalone verification script for the compressed/no_examples JSON-parse-failure fix.

Loads Llama-3.3-70B-Instruct once and re-runs the "compressed" and "no_examples"
prompt variants over the full 300-sample prompt-robustness benchmark, using the
exact same harness settings (batch_size=100, max_tokens=512, temperature=0.0,
seed=42) as judge/scripts/prompt_perturbation.py, so results are directly
comparable to prompt_robustness_outputs/{compressed,no_examples}/llama.json.

Does NOT touch the original prompt_robustness_outputs/ files — writes to
prompt_robustness_outputs_fixed/ instead so the original (broken) run is
preserved for comparison.
"""

import os
import re
import sys
import json
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams
from dotenv import load_dotenv

load_dotenv()
if os.getenv("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.getenv("HF_TOKEN"))

from prompts import build_prompt


# Exact copy of judge/scripts/prompt_perturbation.py's parser, so results are
# comparable to the original (broken) prompt_robustness_outputs/ run.
def extract_json(text: str) -> Optional[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    if "</think>" in text:
        post_think = text.split("</think>")[-1].strip()
        try:
            return json.loads(post_think)
        except Exception:
            match = re.search(r"\{.*\}", post_think, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
    matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group(0))
        except Exception:
            pass
    return None


def safe_get(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

REPO = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_JSON = os.path.join(REPO, "eval_cache", "prompt_robustness_benchmark.json")
OUT_ROOT = os.path.join(REPO, "prompt_robustness_outputs_fixed")

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
VARIANTS = ["compressed", "no_examples"]


def build(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def main():
    with open(BENCHMARK_JSON) as f:
        benchmark = json.load(f)
    pairs = [(x["reference"], x["hypothesis"]) for x in benchmark]
    print(f"Loaded {len(pairs)} samples.")

    build("Loading Llama-3.3-70B-Instruct (tp=4)")
    t0 = time.time()
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.85,
        dtype="float16",
        disable_log_stats=True,
        max_model_len=8000,
        enforce_eager=True,
    )
    print(f"Loaded in {time.time()-t0:.1f}s")

    params = SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0, seed=42)

    os.makedirs(OUT_ROOT, exist_ok=True)

    for variant in VARIANTS:
        build(f"Running variant: {variant}")
        prompts = [
            build_prompt(ref.strip(), hyp.strip(), MODEL_NAME, prompt_type=variant)
            for ref, hyp in pairs
        ]

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=params)
        runtime = time.time() - t0
        raw_texts = [o.outputs[0].text if o.outputs else "" for o in outputs]

        results = []
        n_fail = 0
        for idx, raw in enumerate(raw_texts):
            parsed = extract_json(raw)
            if parsed is None:
                n_fail += 1
                results.append({
                    "contextual_status": "Error",
                    "is_critical": None,
                    "meaning_preserved": None,
                    "operational_safety": None,
                    "explanation": "json parse fail",
                    "transcription_errors": [],
                    "raw_output": raw,
                    "sample_index": idx,
                })
            else:
                quality = safe_get(parsed, "transcription_quality", "other")
                results.append({
                    "contextual_status": quality,
                    "is_critical": quality == "Critical_Errors",
                    "meaning_preserved": safe_get(parsed, "meaning_preserved"),
                    "operational_safety": safe_get(parsed, "operational_safety"),
                    "explanation": safe_get(parsed, "explanation", ""),
                    "transcription_errors": safe_get(parsed, "transcription_errors", []),
                    "raw_output": raw,
                    "sample_index": idx,
                })

        print(f"[{variant}] fails: {n_fail}/{len(results)} ({n_fail/len(results)*100:.1f}%)  runtime: {runtime:.1f}s")

        out_dir = os.path.join(OUT_ROOT, variant)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "llama.json")
        with open(out_path, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "prompt_variant": variant,
                "num_samples": len(results),
                "runtime_sec": round(runtime, 2),
                "records": results,
            }, f, indent=2)
        print(f"Saved -> {out_path}")

    build("DONE")


if __name__ == "__main__":
    main()
