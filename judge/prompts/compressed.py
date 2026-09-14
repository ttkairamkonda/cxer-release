COMPRESSED_COT_PROMPT = """\
You are an expert ATC evaluator. Compare reference vs transcription and detect ONLY operationally significant meaning-changing errors. Ignore filler words and equivalent ICAO variants.

REFERENCE: {reference}
TRANSCRIPTION: {transcription}

--- ICAO PHONETICS ---
1=one/wun, 2=two/too, 3=tree/three, 4=fower/four, 5=fife/five, 6=six, 7=seven, 8=ait/eight, 9=niner/nine, 0=zero
A=alfa/alpha, B=bravo, C=charlie, D=delta, E=echo, F=foxtrot, G=golf, H=hotel, I=india, J=juliett, K=kilo, L=lima, M=mike, N=november, O=oscar, P=papa, Q=quebec, R=romeo, S=sierra, T=tango, U=uniform, V=victor, W=whiskey, X=xray, Y=yankee, Z=zulu

--- CRITICAL ERROR TYPES ---
Callsign, runway, altitude, heading, frequency, direction (left/right), clearance, navigation, weather.

Differences are CRITICAL if they change any of the above.

--- IGNORE ---
Filler words (roger, wilco, copy, etc.), spacing/tokenization, and ICAO equivalents (tree=3, fife=5, niner=9, alfa=alpha). Also ignore wording/grammar changes that preserve meaning.

--- RULES ---
1. Normalize ICAO + equivalents.
2. Extract key entities (callsign, runway, altitude, heading, frequency, direction, clearance, navigation, weather).
3. Compare meaning.
4. Mark ONLY non-equivalent differences as errors.
5. Assign entity_type to highest priority category.
6. Set:
   - Critical error exists -> "Critical_Errors", meaning_preserved=false
   - Else -> "Equivalent", meaning_preserved=true
7. Set safety:
   - runway/altitude/clearance/direction -> unsafe
   - callsign/frequency/heading -> potentially_unsafe
   - others -> safe

Apply rules 1-7 silently. Do not write out your reasoning, do not narrate the steps in prose, and do not restate the rules — go directly from the input to the final JSON object.

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
      "reference_value": "exact phrase",
      "transcribed_value": "exact phrase",
      "safety_risk": "high|medium|low",
      "explanation": "one sentence"
    }
  ]
}

If no critical errors: return empty transcription_errors array.
"""