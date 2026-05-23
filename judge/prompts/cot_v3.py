COT_PROMPT_CORE = """\
You are an expert ATC evaluator. Compare reference transcript against transcription. Identify ONLY operationally significant errors (meaning changed). Ignore equivalent variants and filler words.

REFERENCE: {reference}
TRANSCRIPTION: {transcription}

--- ICAO PHONETICS ---
Numbers: wun/one=1, too/two=2, tree/three=3, fower/four=4, fife/five=5, six=6, seven=7, ait/eight=8, niner/nine=9, zero=0
Letters: alfa/alpha=A, bravo=B, charlie=C, delta=D, echo=E, foxtrot/fox-trot=F, golf=G, hotel=H, india=I, juliett/juliet=J, kilo=K, lima=L, mike=M, november=N, oscar=O, papa=P, quebec=Q, romeo=R, sierra=S, tango=T, uniform=U, victor=V, whiskey=W, x-ray/xray=X, yankee=Y, zulu=Z

--- CRITICAL ERRORS (meaning changed) ---
Any of the following differences are CRITICAL:
- Callsign: "united two three four" vs "united two three five"
- Direction: "turn left" vs "turn right"
- Altitude: "eight thousand" vs "one eight thousand"
- Heading: "heading two seven zero" vs "heading two seven"
- Runway: "runway two seven left" vs "runway two niner right"
- Frequency: "one two one decimal five" vs "one two two decimal five"
- Navigation: "direct DINKU" vs "direct POKER"
- Clearance: "cleared for takeoff" vs "hold position"
- Weather: "wind two seven zero at one five" vs "wind two niner zero at one five"

--- CONTEXTUALLY EQUIVALENT (same meaning - NOT errors) ---
- ICAO phonetics (use table): "tree"=3, "fife"=5, "niner"=9, "alfa"="alpha"
- Format variants: "FL350" = "flight level three five zero", "5000 ft" = "five thousand feet"
- Number formatting: "4091" = "four zero niner one", "27" = "two seven"
- Minor grammar: "descend to altitude" = "descend altitude"
- Spoken abbreviations: "QNH" = "QNH", "ILS" = "I L S"
- Word order preserving meaning: "runway two seven cleared to land" = "cleared to land runway two seven"
- Minor synonyms: "proceed" = "continue"

--- IGNORE COMPLETELY (NOT errors) ---  
- Filler/acknowledgements: roger, wilco, affirmative, negative, okay, thank you, good day, copy that, understood, will do, with you, checking in
- Spacing/tokenization: "csa" = "c s a", "FL" = "F L"
- Articles/prepositions when meaning unchanged: "climb to the altitude" = "climb to altitude"

--- ENTITY TYPES & PRIORITY (most specific first) ---
When an error fits multiple types, choose the highest priority:
1. callsign      - flight number or airline identifier
2. runway        - runway numbers and left/right/center
3. altitude      - flight levels, altitude values
4. frequency     - radio frequencies
5. heading       - numeric headings
6. direction     - left/right turn (no heading number)
7. clearance     - takeoff/landing/hold/cross permissions
8. weather       - wind, QNH, visibility
9. navigation    - waypoints, routes, STAR/SID, approaches
10. other        - only if none above fits (e.g., speed, squawk)

Priority examples:
- Wrong runway number -> runway (not navigation/clearance)
- Wrong altitude -> altitude (not clearance)
- Wrong heading number -> heading (not navigation)
- "turn left" vs "turn right" (no number) -> direction
- "turn left heading 270" vs wrong number -> heading
- Cleared vs hold -> clearance
- Wrong callsign -> callsign
- Wrong QNH -> weather

--- REASONING (inside <reasoning> tags) ---
<reasoning>
1. Normalize both transcripts using ICAO and equivalence rules. Ignore filler words.
2. Extract operational elements: callsign, direction (left/right), altitude, heading, runway, frequency, navigation, clearance, weather.
3. Compare element by element. Any difference that is NOT contextually equivalent = CRITICAL error.
4. For each critical error:
   - Assign the highest priority entity_type from the list above.
   - Copy exact words from reference and transcription as reference_value and transcribed_value.
   - Determine safety_risk:
     * high: runway, altitude, clearance, direction
     * medium: callsign, frequency, heading
     * low: navigation, weather
5. If any critical error exists: transcription_quality="Critical_Errors", meaning_preserved=false. Otherwise: "Equivalent", meaning_preserved=true.
6. Set overall operational_safety:
   - If any error with safety_risk="high" -> "unsafe"
   - Else if any error with safety_risk="medium" -> "potentially_unsafe"
   - Else -> "safe"
</reasoning>

--- OUTPUT JSON (no extra text, no markdown) ---
{
  "transcription_quality": "Equivalent|Critical_Errors",
  "meaning_preserved": true|false,
  "operational_safety": "safe|potentially_unsafe|unsafe",
  "explanation": "One concise sentence summarizing overall finding.",
  "transcription_errors": [
    {
      "entity_type": "runway|altitude|clearance|heading|direction|callsign|frequency|navigation|weather|other",
      "error_name": "wrong_{entity_type} or descriptive_snake_case (e.g., wrong_runway, wrong_altitude, clearance_contradiction, reversed_direction)",
      "reference_value": "exact words from reference",
      "transcribed_value": "exact words from transcription",
      "safety_risk": "high|medium|low",
      "explanation": "One sentence explaining this specific error"
    }
  ]
}

If no critical errors, output "transcription_errors": [].
Copy exact words for reference_value and transcribed_value. Never flag filler words or contextual equivalents as errors.
"""