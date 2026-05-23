from .base import BASE_PROMPT_CORE
from .cot import COT_PROMPT_CORE
from .cot_v3 import COT_PROMPT_CORE as COT_V3_PROMPT_CORE
from .compressed import COMPRESSED_COT_PROMPT
from .no_examples import NO_EXAMPLES_COT_PROMPT
from .lenient import LENIENT_COT_PROMPT

_REGISTRY = {
    "base":        BASE_PROMPT_CORE,
    "cot":         COT_PROMPT_CORE,
    "cot_v3":      COT_V3_PROMPT_CORE,
    "compressed":  COMPRESSED_COT_PROMPT,
    "no_examples": NO_EXAMPLES_COT_PROMPT,
    "lenient":     LENIENT_COT_PROMPT,
}

def build_prompt(reference, transcription, model_name, prompt_type="cot_v3"):
    template = _REGISTRY.get(prompt_type)
    if template is None:
        raise ValueError(f"Unknown prompt_type '{prompt_type}'. Available: {list(_REGISTRY)}")

    # Use replace() instead of .format() to avoid conflicts with
    # literal JSON curly braces inside the prompt templates
    core = template.replace("{reference}", reference.strip())
    core = core.replace("{transcription}", transcription.strip())

    model = model_name.lower()

    if "mistral" in model:
        return f"[INST] {core} [/INST]"
    elif "gemma" in model:
        return f"<start_of_turn>user\n{core}<end_of_turn>\n<start_of_turn>model\n"
    elif "llama" in model:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    elif "qwen" in model:
        return f"<|im_start|>user\n{core}<|im_end|>\n<|im_start|>assistant\n"
    elif "deepseek" in model:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    else:
        return core
