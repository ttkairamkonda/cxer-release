NO_EXAMPLES_COT_PROMPT = """\
You are an expert ATC evaluator. Compare reference vs transcription and identify ONLY operationally significant meaning-changing differences. Ignore fillers and context-preserving variants.

REFERENCE: {reference}
TRANSCRIPTION: {transcription}

--- ICAO NORMALIZATION ---
Convert spoken ICAO words into standard digits/letters where applicable and treat equivalent pronunciations as identical.

--- ERROR DEFINITIONS ---
A critical error is any difference that changes operational meaning in:
callsign, runway, altitude, heading, frequency, direction (left/right), clearance, navigation, weather.

--- IGNORE ---
Ignore fillers, acknowledgements, spacing differences, and ICAO phonetic variants. Ignore any wording differences that preserve meaning.

--- DECISION RULE ---
1. Normalize both inputs.
2. Extract operational entities: callsign, runway, altitude, heading, frequency, direction, clearance, navigation, weather.
3. Compare semantic meaning at entity level.
4. Mark differences as critical ONLY if meaning changes.
5. Assign each error to the highest priority entity type.
6. Determine:
   - Critical error exists → transcription_quality = "Critical_Errors", meaning_preserved = false
   - Else → "Equivalent", meaning_preserved = true

--- SAFETY RULE ---
Assign operational_safety:
- runway, altitude, clearance, direction → unsafe
- callsign, frequency, heading → potentially_unsafe
- navigation, weather → safe

Apply the decision rule and safety rule silently. Do not write out your reasoning, do not narrate the steps in prose — go directly from the input to the final JSON object.

--- OUTPUT: RESPOND WITH THE JSON OBJECT ONLY ---
No preamble, no markdown fences, no text before or after the object.
{
  "transcription_quality": "Equivalent|Critical_Errors",
  "meaning_preserved": true|false,
  "operational_safety": "safe|potentially_unsafe|unsafe",
  "explanation": "one sentence summary",
  "transcription_errors": [
    {
      "entity_type": "callsign|runway|altitude|heading|frequency|direction|clearance|navigation|weather|other",
      "error_name": "snake_case_label",
      "reference_value": "exact phrase from reference",
      "transcribed_value": "exact phrase from transcription",
      "safety_risk": "high|medium|low",
      "explanation": "one sentence"
    }
  ]
}

If no critical errors exist, return an empty transcription_errors list.
"""