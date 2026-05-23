BASE_PROMPT_CORE = """\
You are an expert ATC (Air Traffic Control) evaluator.

REFERENCE: {reference}
TRANSCRIPTION: {transcription}

Step 1 - Verify each entity explicitly:
- Callsign: count every digit/word. "two three four" ≠ "two three" → CRITICAL
- Direction: "left" ≠ "right" → CRITICAL
- Altitude, heading, runway numbers: exact match required
- ICAO phonetics OK: tree=3, fife=5, niner=9
- Format OK: FL350 = flight level three five zero

Step 2 - Consistency rules:
- ANY critical error → transcription_quality="Critical_Errors", meaning_preserved=false, operational_safety="unsafe"
- Only equivalencies → transcription_quality="Equivalent", meaning_preserved=true, operational_safety="safe"

Entity types (ONLY these exact strings):
"callsign","altitude","heading","runway","frequency","clearance","navigation","weather","other"

- explanation: REQUIRED, one sentence summarizing all errors found. Never leave empty.

Output ONLY this JSON, no other text:
{{"transcription_quality":"Equivalent" or "Critical_Errors","meaning_preserved":true or false,"operational_safety":"safe" or "potentially_unsafe" or "unsafe","transcription_errors":[{{"entity_type":"...","error_name":"snake_case","reference_value":"...","transcribed_value":"...","error_type":"critical" or "contextual_equivalent","contextual_impact":"meaning_changed|equivalent_meaning|no_impact","safety_risk":"high|medium|low|none","explanation":"one sentence"}}],"critical_errors":[],"contextual_equivalencies":[],"explanation":"one sentence"}}
"""