LENIENT_COT_PROMPT = """\
You are an expert ATC evaluator. Compare reference and transcription and determine whether the meaning is preserved or changed. Focus only on whether the two utterances would lead to different operational understanding.

REFERENCE: {reference}
TRANSCRIPTION: {transcription}

--- TASK ---
Determine if the transcription preserves the same meaning as the reference in an aviation context.

Ignore minor wording differences. Focus on whether any change could alter instructions, intent, or safety-critical meaning.

--- DECISION RULE ---
1. Read both utterances.
2. Compare overall meaning in context.
3. Decide if any meaningful difference exists:
   - If meaning is preserved → "Equivalent"
   - If meaning changes → "Critical_Errors"
4. Infer operational risk based on severity of meaning change.

--- OUTPUT RULES ---
- Do NOT rely on predefined categories.
- Do NOT assume structured error types.
- Do NOT normalize phonetics explicitly.
- Use only semantic judgment.

--- OUTPUT (STRICT JSON ONLY) ---
{
  "transcription_quality": "Equivalent|Critical_Errors",
  "meaning_preserved": true|false,
  "operational_safety": "safe|potentially_unsafe|unsafe",
  "explanation": "one sentence summary",
  "transcription_errors": []
}

If no meaningful error exists, return an empty transcription_errors list.
"""