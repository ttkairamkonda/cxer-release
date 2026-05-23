"""
ATC Phraseology Evaluator — Batched Version
Key performance improvements over previous version:
  - Batch all prompts per file in a single llm.generate() call (10-50x faster)
  - Optional cross-file parallelism with ThreadPoolExecutor
  - Chunk-based batching with configurable BATCH_SIZE to manage GPU memory
  - Progress tracking with tqdm
  - Resume support: skips output files that already exist
  - Same output schema as before — fully backward compatible
"""

# CUDA_VISIBLE_DEVICES=4,5,6,7

import os
import re
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm import LLM, SamplingParams
from jiwer import wer

from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

login(token=os.getenv("HF_TOKEN"))


try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not installed — install with: pip install tqdm")

# MODEL_NAME = "google/gemma-2-9b-it"
# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

PROMPT_TYPE = "cot_v3"
# ─────────────────────────────────────────────
# 1.  MODEL INIT
# ─────────────────────────────────────────────


def load_model(
    # model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    # model_name : str = "EleutherAI/pythia-6.9b",
    # model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    # model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    model_name: str = MODEL_NAME,
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 8000,
) -> LLM:
    import torch
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="float16",
        disable_log_stats=True,
        compilation_config={"level": 0},
        max_model_len=max_model_len,
    )
    print("Model loaded successfully!")
    return llm


# ─────────────────────────────────────────────
# 2.  PROMPT TEMPLATE
# ─────────────────────────────────────────────
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import build_prompt

# ─────────────────────────────────────────────
# 3.  ROBUST JSON EXTRACTION
# ─────────────────────────────────────────────

def extract_json(raw_text: str) -> Optional[dict]:
    raw_text = re.sub(r"<reasoning>.*?</reasoning>", "", raw_text, flags=re.DOTALL).strip()

    # Try clean parse first
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Try markdown block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try first complete JSON object
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try repairing truncated JSON — add closing braces
    for raw in [raw_text, brace_match.group(0) if brace_match else ""]:
        if not raw:
            continue
        for suffix in ["}", "}]}", "}]}\n}"]:
            try:
                return json.loads(raw + suffix)
            except json.JSONDecodeError:
                continue

    return None


def safe_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


# ─────────────────────────────────────────────
# 4.  EVALUATOR CLASS — BATCH-FIRST DESIGN
# ─────────────────────────────────────────────

class ATCEvaluator:
    """
    Batched ATC transcription evaluator.
    
    Core change: evaluate_batch() sends ALL prompts in one llm.generate() call.
    vLLM processes them in parallel across GPU workers — this is the main speedup.
    
    BATCH_SIZE controls how many prompts are sent per generate() call.
    Larger = better GPU utilization but more peak memory.
    Start with 100, increase if you have headroom.
    """

    VALID_ENTITY_TYPES = {
        "callsign", "altitude", "heading", "runway",
        "frequency", "clearance", "navigation", "weather", "other"
    }

    # def __init__(
    #     self,
    #     llm: LLM,
    #     max_tokens: int = 1024,         # ← increased from 600; JSON responses need room
    #     temperature: float = 0.0,
    #     batch_size: int = 100,
    # ):
    def __init__(self, llm, model_name: str, max_tokens=2048, temperature=0.0, batch_size=100, prompt_type=PROMPT_TYPE):
        self.model_name = model_name
        self.llm = llm
        self.batch_size = batch_size
        self.prompt_type = prompt_type
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(f"ATCEvaluator ready (batch_size={batch_size})")

    # ── batch LLM call — THE KEY PERFORMANCE CHANGE ───────────────────────────

    def _call_llm_batch(self, prompts: List[str]) -> List[str]:
        """
        Send all prompts in one vLLM generate() call.
        Each prompt is suffixed with a '{' primer so the model is forced
        to continue as JSON rather than repeating the instruction text.
        """
        # Append '{' to each prompt — the model must continue from there,
        # so the first token it generates is always inside a JSON object.
        # primed = [p + "\n{" for p in prompts]
        primed = prompts
        outputs = self.llm.generate(primed, sampling_params=self.sampling_params)
        # raw_texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
        # print("RAW:", raw_texts[0][:500])
        # return raw_texts
        return [o.outputs[0].text if o.outputs else "" for o in outputs]


    # ── evaluate a list of (reference, transcription) pairs ───────────────────

    def evaluate_batch(
        self,
        pairs: List[Tuple[str, str]],       # list of (reference, transcription)
        start_index: int = 0,               # for sample_index tracking
        label: str = "",                    # for progress bar label
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of ref/hyp pairs in batched chunks.
        Returns results in the same order as input pairs.
        """
        results = []
        total = len(pairs)

        iterator = range(0, total, self.batch_size)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc=label or "Evaluating", unit="batch")

        for chunk_start in iterator:
            chunk = pairs[chunk_start : chunk_start + self.batch_size]

            # Build prompts for the whole chunk
            prompts = []
            valid_mask = []  # track which samples had empty input
            for ref, hyp in chunk:
                if not ref.strip() or not hyp.strip():
                    prompts.append(None)
                    valid_mask.append(False)
                else:
                    prompts.append(build_prompt(ref.strip(), hyp.strip(), self.model_name, prompt_type=self.prompt_type))

                    # prompts.append(COMPARISON_PROMPT_TEMPLATE.format(
                    #     reference=ref.strip(),
                    #     transcription=hyp.strip(),
                    # ))
                    valid_mask.append(True)

            # One batched LLM call for all valid prompts in the chunk
            valid_prompts = [p for p in prompts if p is not None]
            t0 = time.time()
            raw_outputs = self._call_llm_batch(valid_prompts) if valid_prompts else []
            batch_time = time.time() - t0
            per_sample_time = batch_time / max(len(valid_prompts), 1)

            # Map outputs back to their original positions
            output_iter = iter(raw_outputs)
            for local_idx, (is_valid, (ref, hyp)) in enumerate(zip(valid_mask, chunk)):
                sample_index = start_index + chunk_start + local_idx

                if not is_valid:
                    results.append(self._error_result("Empty input", 0.0, sample_index))
                    continue

                raw = next(output_iter)
                parsed = extract_json(raw)

                if parsed is None:
                    results.append(self._error_result(
                        f"JSON parse failed. Raw:\n{raw[:300]}",
                        per_sample_time,
                        sample_index,
                    ))
                else:
                    results.append(self._build_result(parsed, per_sample_time, sample_index))

        return results

    # ── single-sample shim (backward compat) ──────────────────────────────────

    def evaluate_single(
        self,
        reference: str,
        transcription: str,
        sample_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.evaluate_batch([(reference, transcription)], start_index=sample_index or 0)[0]

    # ── result builders ────────────────────────────────────────────────────────

    def _build_result(self, parsed: dict, inference_time: float, sample_index) -> Dict[str, Any]:
        errors = safe_get(parsed, "transcription_errors", []) or []

        # normalize entity_type
        for err in errors:
            if isinstance(err, dict):
                if err.get("entity_type") not in self.VALID_ENTITY_TYPES:
                    err["entity_type"] = "other"

        quality = safe_get(parsed, "transcription_quality", "other")
        if quality not in ("Equivalent", "Critical_Errors"):
            quality = "other"

        return {
            "contextual_status":    quality,
            "meaning_preserved":    safe_get(parsed, "meaning_preserved"),
            "operational_safety":   safe_get(parsed, "operational_safety"),
            "explanation":          safe_get(parsed, "explanation", ""),
            "transcription_errors": errors,
            "inference_time":       round(inference_time, 4),
            "sample_index":         sample_index,
        }

    @staticmethod
    def _error_result(msg: str, inference_time: float, sample_index) -> Dict[str, Any]:
        return {
            "contextual_status":    "Error",
            "meaning_preserved":    None,
            "operational_safety":   None,
            "explanation":          msg,
            "transcription_errors": [],
            "inference_time":       round(inference_time, 4),
            "sample_index":         sample_index,
        }


# ─────────────────────────────────────────────
# 5.  FILE / BATCH EVALUATION
# ─────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> Optional[float]:
    try:
        return round(wer(reference, hypothesis), 4)
    except Exception:
        return None


def evaluate_single_file(
    input_json_path: str,
    output_json_path: str,
    evaluator: ATCEvaluator,
    max_samples: Optional[int] = None,
    skip_existing: bool = True,          # ← resume support
) -> None:
    """
    Evaluate all samples in one JSON file using batched inference.
    
    skip_existing=True: if the output file already exists, skip this file.
    This lets you resume a crashed run without re-processing completed files.
    """

    # ── resume support ──────────────────────────────────────────────────────
    if skip_existing and os.path.exists(output_json_path):
        print(f"  ⏭  Skipping (already done): {output_json_path}")
        return

    with open(input_json_path) as f:
        data = json.load(f)

    # Handle both formats
    records = data["samples"] if isinstance(data, dict) else data

    if not isinstance(records, list):
        raise ValueError(f"{input_json_path} must contain a JSON list under 'samples'")

    if max_samples is not None:
        records = records[:max_samples]

    # ── build all ref/hyp pairs upfront ────────────────────────────────────
    pairs = [(r["reference"], r["hypothesis"]) for r in records]
    all_refs = [r["reference"] for r in records]
    all_hyps = [r["hypothesis"] for r in records]

    file_label = os.path.basename(input_json_path)

    # ── ONE batched evaluate call for the whole file ────────────────────────
    t0 = time.time()
    results = evaluator.evaluate_batch(pairs, start_index=0, label=file_label)
    total_time = time.time() - t0

    print(f"  ⏱  {file_label}: {len(pairs)} samples in {total_time:.1f}s "
          f"({total_time/max(len(pairs),1):.3f}s/sample)")

    # ── merge results back into records ────────────────────────────────────
    evaluated = []
    for record, result in zip(records, results):
        result["WER"] = compute_wer(record["reference"], record["hypothesis"])
        evaluated.append({**record, **result})

    corpus_wer_val = round(wer(all_refs, all_hyps), 4) if all_refs else None

    output = {
        "num_samples_evaluated": len(evaluated),
        "corpus_wer":            corpus_wer_val,
        "total_inference_time":  round(total_time, 2),
        "records":               evaluated,
    }

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved → {output_json_path}")


def evaluate_all_files(
    input_root: str,
    output_root: str,
    evaluator: ATCEvaluator,
    max_samples: Optional[int] = None,
    skip_existing: bool = True,
    max_file_workers: int = 1,           # >1 = parallel file processing (use carefully)
) -> None:
    """
    Walk all subdirectories under input_root and evaluate every JSON file.
    
    max_file_workers=1  : process files sequentially (default, safest for GPU)
    max_file_workers>1  : process multiple files concurrently — only do this
                          if you have spare CPU threads and the bottleneck is
                          I/O, not GPU. For GPU-bound work keep this at 1.
    """
    os.makedirs(output_root, exist_ok=True)

    # Collect all (input_path, output_path) pairs
    file_pairs = []
    entries = sorted(os.listdir(input_root))

    for entry in entries:
        entry_path = os.path.join(input_root, entry)

        # Case 1: direct JSON files inside input_root
        if os.path.isfile(entry_path) and entry.endswith(".json"):
            file_pairs.append((
                entry_path,
                os.path.join(output_root, entry),
            ))

        # Case 2: subdirectories containing JSON files
        elif os.path.isdir(entry_path):
            output_dataset_dir = os.path.join(output_root, entry)
            os.makedirs(output_dataset_dir, exist_ok=True)

            json_files = sorted(
                f for f in os.listdir(entry_path)
                if f.endswith(".json")
            )

            for file_name in json_files:
                file_pairs.append((
                    os.path.join(entry_path, file_name),
                    os.path.join(output_dataset_dir, file_name),
                ))

    print(f"\nTotal files to process: {len(file_pairs)}")

    if max_file_workers <= 1:
        # Sequential (recommended for GPU-bound workloads)
        for in_path, out_path in file_pairs:
            print(f"\n▶ {in_path}")
            evaluate_single_file(
                input_json_path=in_path,
                output_json_path=out_path,
                evaluator=evaluator,
                max_samples=max_samples,
                skip_existing=skip_existing,
            )
    else:
        # Parallel file processing (I/O overlap, not for GPU parallelism)
        def _task(args):
            in_path, out_path = args
            evaluate_single_file(
                input_json_path=in_path,
                output_json_path=out_path,
                evaluator=evaluator,
                max_samples=max_samples,
                skip_existing=skip_existing,
            )

        with ThreadPoolExecutor(max_workers=max_file_workers) as pool:
            futures = {pool.submit(_task, pair): pair for pair in file_pairs}
            for future in as_completed(futures):
                in_path, _ = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ERROR in {in_path}: {e}")


# ─────────────────────────────────────────────
# 6.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":


    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    MODELS = [
        ("Qwen/Qwen2.5-72B-Instruct",                 4, os.path.join(_REPO, "annotations", "qwen_72b")),
        ("meta-llama/Llama-3.3-70B-Instruct",         4, os.path.join(_REPO, "annotations", "llama_70b")),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 4, os.path.join(_REPO, "annotations", "deepseek_70b")),
    ]

    for model_name, tensor_parallel_size, output_root in MODELS:
        print(f"\n{'='*60}")
        print(f"Running model: {model_name}")
        print(f"{'='*60}")

        llm = load_model(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
        )

        evaluator = ATCEvaluator(
            llm,
            model_name=model_name,
            max_tokens=2048,
            temperature=0.0,
            batch_size=200,
            prompt_type=PROMPT_TYPE,
        )


        # Quick smoke test — uncomment to verify correctness before full run
        ## Example 1
        # ref = "United two three four heavy climb and maintain flight level three five zero, turn left heading two seven zero"
        # hyp = "United two tree heavy climb FL350, turn right heading two seven zero"

        ## Example 2
        # ref = "Speedbird four five six descend to altitude eight thousand feet, QNH one zero one three"
        # hyp = "Speedbird four five six descend eight thousand, QNH one zero one three"

        # # Example 3
        # ref = "Speedbird four five six descend to altitude eight thousand feet, QNH one zero one three"
        # hyp = "Speedbird four five six descend to altitude one eight thousand feet, QNH one zero one three"
        
        # prompt = COMPARISON_PROMPT_TEMPLATE.format(reference=ref, transcription=hyp)
        # print(prompt)
        # result = evaluator.evaluate_single(ref, hyp, sample_index=0)
        # print(json.dumps(result, indent=2))
        
        evaluate_all_files(
            input_root=os.path.join(_REPO, "data", "raw"),
            output_root=output_root,
            evaluator=evaluator,
            # max_samples=10,
            skip_existing=True,
        )

        # explicitly destroy before loading next model
        del llm
        del evaluator
        import torch
        torch.cuda.empty_cache()
        print(f"Done: {model_name}")