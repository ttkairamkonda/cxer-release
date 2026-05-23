"""
ATC Prompt Robustness Evaluation Pipeline (FIXED)
==================================================
Drop-in robust version with:
- Prompt variant validation
- Safe batching
- Stable JSON extraction
- Consistent outputs
- Dispatcher __init__.py instructions included as comment
- Loop order fixed: model outer, prompt inner (load each model once)
"""

import os
import re
import sys
import json
import time
import torch
from typing import Optional, List, Dict, Any, Tuple

from vllm import LLM, SamplingParams
from huggingface_hub import login
from dotenv import load_dotenv

# ============================================================
# ENV
# ============================================================

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================
# PROMPTS
# ============================================================

# NOTE: Your prompts/ directory must have an __init__.py that dispatches
# by prompt_type. It should look like this:
#
#   from .cot_v3 import build_prompt as _cot_v3
#   from .compressed import build_prompt as _compressed
#   from .no_examples import build_prompt as _no_examples
#   from .strict import build_prompt as _strict
#   from .lenient import build_prompt as _lenient
#
#   _REGISTRY = {
#       "cot_v3":      _cot_v3,
#       "compressed":  _compressed,
#       "no_examples": _no_examples,
#       "strict":      _strict,
#       "lenient":     _lenient,
#   }
#
#   def build_prompt(ref, hyp, model_name, prompt_type="cot_v3"):
#       fn = _REGISTRY.get(prompt_type)
#       if fn is None:
#           raise ValueError(f"Unknown prompt_type '{prompt_type}'. Available: {list(_REGISTRY)}")
#       return fn(ref, hyp, model_name)

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from prompts import build_prompt

# ============================================================
# CONFIG
# ============================================================

PROMPT_VARIANTS = ["cot_v3", "compressed", "no_examples", "lenient"]

MODEL_CONFIGS = [
    # ("Qwen/Qwen2.5-72B-Instruct",               4, "qwen"),
    ("meta-llama/Llama-3.3-70B-Instruct",        4, "llama"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 4, "deepseek"),
]

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCHMARK_JSON = os.path.join(_REPO, "eval_cache", "prompt_robustness_benchmark.json")
OUTPUT_ROOT    = os.path.join(_REPO, "prompt_robustness_outputs")

# ============================================================
# MODEL LOADER
# ============================================================

def load_model(model_name: str, tp_size: int) -> LLM:
    torch.cuda.empty_cache()

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.85,
        dtype="float16",
        disable_log_stats=True,
        max_model_len=8000,
        # Force chat template application
        enforce_eager=True,
    )
    print(f"\nLoaded model: {model_name}")
    return llm

# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text: str) -> Optional[dict]:
    # Strip DeepSeek-R1 chain-of-thought blocks (<think> and <reasoning>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt 2: JSON inside a code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Attempt 3: first {...} in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Attempt 4: DeepSeek sometimes puts JSON after a blank line post-think
    # Try extracting everything after the last </think> if present
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

# ============================================================
# SAFE GET
# ============================================================

def safe_get(d: Any, k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

# ============================================================
# EVALUATOR
# ============================================================

class ATCEvaluator:

    VALID_PROMPTS = set(PROMPT_VARIANTS)

    def __init__(
        self,
        llm: LLM,
        model_name: str,
        batch_size: int = 100,
        max_tokens: int = 512,
        temperature: float = 0.0,
        prompt_type: str = "cot_v3",
    ):
        if prompt_type not in self.VALID_PROMPTS:
            print(f"[WARNING] Unknown prompt: '{prompt_type}'. Falling back to 'cot_v3'")
            prompt_type = "cot_v3"

        self.llm        = llm
        self.model_name = model_name
        self.batch_size = batch_size
        self.prompt_type = prompt_type
        if "deepseek" in model_name.lower():
            self.params = SamplingParams(
                max_tokens=2048,  # needs more tokens for think + JSON
                temperature=0.0,
                top_p=1.0,
                seed=42,
                stop=["<｜end▁of▁sentence｜>"],
            )
        else:
            self.params = SamplingParams(
                max_tokens=512,
                temperature=0.0,
                top_p=1.0,
                seed=42,
            )
        # self.params = SamplingParams(
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=1.0,
        #     seed=42,
        # )
        

    def _call(self, prompts: List[str]) -> List[str]:
        # DeepSeek-R1-Distill needs explicit chat formatting
        if "deepseek" in self.model_name.lower():
            formatted = [
                f"<｜begin▁of▁sentence｜><｜User｜>{p}<｜Assistant｜>"
                for p in prompts
            ]
        else:
            formatted = prompts

        outputs = self.llm.generate(formatted, sampling_params=self.params)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]

    def evaluate_batch(
        self,
        pairs: List[Tuple[str, str]],
        start_index: int = 0,
        label: str = "",
    ) -> List[dict]:

        results = []
        iterator = range(0, len(pairs), self.batch_size)

        if HAS_TQDM:
            iterator = tqdm(iterator, desc=label)

        for start in iterator:

            chunk = pairs[start : start + self.batch_size]

            prompts = []
            mask    = []

            for ref, hyp in chunk:
                if not ref.strip() or not hyp.strip():
                    prompts.append(None)
                    mask.append(False)
                    continue

                prompts.append(
                    build_prompt(
                        ref.strip(),
                        hyp.strip(),
                        self.model_name,
                        prompt_type=self.prompt_type,
                    )
                )
                mask.append(True)

            valid       = [p for p in prompts if p is not None]
            raw_outputs = self._call(valid) if valid else []
            out_iter    = iter(raw_outputs)

            for i, ((ref, hyp), ok) in enumerate(zip(chunk, mask)):
                idx = start_index + start + i

                if not ok:
                    results.append(self._error("empty input", idx))
                    continue

                raw    = next(out_iter)
                if self.model_name and "deepseek" in self.model_name.lower():
                    print(f"\n[DEBUG DeepSeek raw output[:500]]:\n{raw[:500]}\n")
                parsed = extract_json(raw)

                if parsed is None:
                    results.append(self._error("json parse fail", idx, raw))
                else:
                    results.append(self._build(parsed, raw, idx))

        return results

    def _build(self, parsed: dict, raw: str, idx: int) -> dict:
        quality     = safe_get(parsed, "transcription_quality", "other")
        is_critical = quality == "Critical_Errors"

        return {
            "contextual_status":    quality,
            "is_critical":          is_critical,
            "meaning_preserved":    safe_get(parsed, "meaning_preserved"),
            "operational_safety":   safe_get(parsed, "operational_safety"),
            "explanation":          safe_get(parsed, "explanation", ""),
            "transcription_errors": safe_get(parsed, "transcription_errors", []),
            "raw_output":           raw,
            "sample_index":         idx,
        }

    def _error(self, msg: str, idx: int, raw: str = "") -> dict:
        return {
            "contextual_status":    "Error",
            "is_critical":          None,
            "meaning_preserved":    None,
            "operational_safety":   None,
            "explanation":          msg,
            "transcription_errors": [],
            "raw_output":           raw,
            "sample_index":         idx,
        }

# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check_prompts(model_name: str) -> bool:
    """Verify every prompt variant builds without error before loading any model."""
    sample_ref = "november one two three cleared to land runway two eight left"
    sample_hyp = "november one two three clear to land runway two eight left"

    print("\n=== Prompt variant sanity check ===")
    all_ok = True

    for variant in PROMPT_VARIANTS:
        try:
            p = build_prompt(sample_ref, sample_hyp, model_name, prompt_type=variant)
            print(f"  ✓  {variant:<15} ({len(p)} chars)")
        except Exception as e:
            print(f"  ✗  {variant:<15} ERROR: {e}")
            all_ok = False

    print()
    return all_ok

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Load benchmark
    with open(BENCHMARK_JSON) as f:
        benchmark = json.load(f)

    pairs = [(x["reference"], x["hypothesis"]) for x in benchmark]
    print(f"Loaded {len(pairs)} samples from benchmark.")

    # Sanity-check all prompt variants before touching any GPU
    if not sanity_check_prompts(MODEL_CONFIGS[0][0]):
        sys.exit("Prompt sanity check failed. Fix prompts/__init__.py before continuing.")

    # --- Outer loop: model (load once per model) ---
    for model_name, tp, short in MODEL_CONFIGS:

        # Skip model load entirely if every variant output already exists
        all_done = all(
            os.path.exists(os.path.join(OUTPUT_ROOT, prompt, f"{short}.json"))
            for prompt in PROMPT_VARIANTS
        )
        if all_done:
            print(f"[{short}] All variants already done — skipping model load.")
            continue

        llm = load_model(model_name, tp)

        # --- Inner loop: prompt variant (no model reload) ---
        for prompt in PROMPT_VARIANTS:

            out_dir  = os.path.join(OUTPUT_ROOT, prompt)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{short}.json")

            if os.path.exists(out_path):
                print(f"[{short}] Skipping already-done variant: {prompt}")
                continue

            print(f"\n[{short}] Running variant: {prompt}")

            evaluator = ATCEvaluator(
                llm,
                model_name,
                batch_size=100,
                prompt_type=prompt,
            )

            t0      = time.time()
            results = evaluator.evaluate_batch(pairs, label=f"{short}-{prompt}")
            runtime = time.time() - t0

            final = {
                "model":          model_name,
                "prompt_variant": prompt,
                "num_samples":    len(results),
                "runtime_sec":    round(runtime, 2),
                "records":        results,
            }

            with open(out_path, "w") as f:
                json.dump(final, f, indent=2)

            print(f"[{short}] Saved {out_path}  ({runtime:.1f}s)")

        # Release GPU memory before loading next model
        del llm
        torch.cuda.empty_cache()

    print("\nAll done.")