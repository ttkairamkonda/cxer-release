"""
Recover true JSON-parse failures in the main CxER annotation set (annotations/)
that feeds Table 1 / Table 2 / Table 4 of the paper.

Scope: only records where contextual_status == "Error" AND the ASR hypothesis
is non-empty (a true judge JSON-parse failure). Records with an empty
hypothesis (ASR total failure) are a separate, distinct category and are
intentionally left untouched here.

For each judge, re-runs the exact same cot_v3 prompt through the same model
(fresh inference — the original raw output wasn't stored for these, only a
300-char truncated snippet), then parses with the json_repair-hardened
extractor now in judge/scripts/llm_judge.py. Successfully recovered records
are patched in place; anything still unparseable after retry is left as
Error, untouched.
"""

import os
import re
import sys
import json
import glob
import time
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge"))

from vllm import LLM, SamplingParams
from prompts import build_prompt
from json_repair import repair_json

VALID_ENTITY_TYPES = {
    "callsign", "altitude", "heading", "runway",
    "frequency", "clearance", "navigation", "weather", "other"
}


# Exact copy of the hardened extract_json now in judge/scripts/llm_judge.py,
# inlined to avoid that module's import-time login()/torch side effects.
def extract_json(raw_text: str) -> Optional[dict]:
    raw_text = re.sub(r"<reasoning>.*?</reasoning>", "", raw_text, flags=re.DOTALL).strip()
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    for raw in [raw_text, brace_match.group(0) if brace_match else ""]:
        if not raw:
            continue
        for suffix in ["}", "}]}", "}]}\n}"]:
            try:
                return json.loads(raw + suffix)
            except json.JSONDecodeError:
                continue
    try:
        fixed = repair_json(brace_match.group(0) if brace_match else raw_text)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

REPO = os.path.dirname(os.path.abspath(__file__))

JUDGES = [
    ("annotations/llama_70b",    "meta-llama/Llama-3.3-70B-Instruct"),
    ("annotations/qwen_72b",     "Qwen/Qwen2.5-72B-Instruct"),
    ("annotations/deepseek_70b", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
]

PROMPT_TYPE = "cot_v3"


def collect_failures(judge_dir):
    """Return list of (file_path, record_index_in_list, record_dict) for
    true JSON-parse failures (non-empty hypothesis) in this judge's files."""
    jobs = []
    for path in sorted(glob.glob(os.path.join(judge_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        for i, r in enumerate(data["records"]):
            if r.get("contextual_status") == "Error" and r.get("hypothesis", "").strip():
                jobs.append((path, i, r))
    return jobs


def main():
    for judge_dir, model_name in JUDGES:
        jobs = collect_failures(judge_dir)
        print(f"\n{'='*70}\n{judge_dir} ({model_name}): {len(jobs)} true JSON-parse failures to retry\n{'='*70}")
        if not jobs:
            continue

        t0 = time.time()
        llm = LLM(
            model=model_name,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.85,
            dtype="float16",
            disable_log_stats=True,
            max_model_len=8000,
            enforce_eager=True,
        )
        print(f"Loaded {model_name} in {time.time()-t0:.1f}s")

        params = SamplingParams(max_tokens=2048, temperature=0.0)

        prompts = [
            build_prompt(r["reference"].strip(), r["hypothesis"].strip(), model_name, prompt_type=PROMPT_TYPE)
            for _, _, r in jobs
        ]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=params)
        print(f"Inference done in {time.time()-t0:.1f}s")
        raw_texts = [o.outputs[0].text if o.outputs else "" for o in outputs]

        # Group recovered updates by file so we write each file once.
        by_file = {}
        n_recovered = 0
        for (path, idx, r), raw in zip(jobs, raw_texts):
            parsed = extract_json(raw)
            by_file.setdefault(path, []).append((idx, parsed, raw))
            if parsed is not None:
                n_recovered += 1

        for path, updates in by_file.items():
            with open(path) as f:
                data = json.load(f)
            for idx, parsed, raw in updates:
                rec = data["records"][idx]
                if parsed is None:
                    rec["explanation"] = f"JSON parse failed (retried with full 2048-token budget + json_repair). Raw:\n{raw[:300]}"
                    continue
                quality = safe_get(parsed, "transcription_quality", "other")
                if quality not in ("Equivalent", "Critical_Errors"):
                    quality = "other"
                errors = safe_get(parsed, "transcription_errors", []) or []
                for err in errors:
                    if isinstance(err, dict) and err.get("entity_type") not in VALID_ENTITY_TYPES:
                        err["entity_type"] = "other"
                rec["contextual_status"] = quality
                rec["meaning_preserved"] = safe_get(parsed, "meaning_preserved")
                rec["operational_safety"] = safe_get(parsed, "operational_safety")
                rec["explanation"] = safe_get(parsed, "explanation", "")
                rec["transcription_errors"] = errors
                rec["recovered_via"] = "retry+json_repair"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        print(f"[{judge_dir}] recovered {n_recovered}/{len(jobs)}")

        del llm
        import torch
        torch.cuda.empty_cache()

    print("\nDONE")


if __name__ == "__main__":
    main()
