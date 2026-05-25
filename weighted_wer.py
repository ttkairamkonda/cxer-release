"""
Weighted WER for ATC transcription.

Entity tokens (callsigns, runways, flight levels, altitudes,
frequencies, headings, squawk codes) are penalised with weight W=3.
All other tokens have weight 1.

Rule-based only — no LLM involved.
"""

import re
import json
import glob
import os
import numpy as np
import pandas as pd
import jiwer

# ── Constants ─────────────────────────────────────────────────────────────────

ENTITY_WEIGHT = 3

# ICAO phonetic alphabet words
PHONETIC = {
    "alfa", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliett", "juliet", "kilo", "lima",
    "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra",
    "tango", "uniform", "victor", "whiskey", "xray", "x-ray", "yankee", "zulu",
}

# Common ATC airline callsign words (first word(s) of callsign)
AIRLINE_NAMES = {
    "lufthansa", "airfrance", "air", "swiss", "alitalia", "iberia",
    "ryanair", "ryan", "easyjet", "easy", "british", "speedbird",
    "united", "delta", "american", "emirates", "qatar", "turkish",
    "korean", "japan", "quantas", "qantas", "austrian", "finnair",
    "klm", "tap", "lot", "csa", "adria", "viva", "ltu", "transwede",
    "crossair", "portugalia", "hapag", "hapagloid", "time", "bel",
    "sky", "euro", "nor", "french", "portugalia", "air_portugal",
    "lufthansa", "nor", "hotel", "speed",
}


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def tokenise(text):
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ── Entity span detector ──────────────────────────────────────────────────────

def entity_token_mask(tokens):
    """
    Returns a boolean list of length len(tokens).
    True  → entity token (weight = ENTITY_WEIGHT)
    False → non-entity token (weight = 1)
    """
    n    = len(tokens)
    mask = [False] * n
    i    = 0

    while i < n:
        tok = tokens[i]

        # ── Flight level: "flight level <number>" ────────────────────────────
        if tok == "flight" and i + 2 < n and tokens[i + 1] == "level" \
                and re.fullmatch(r"\d+", tokens[i + 2]):
            mask[i] = mask[i + 1] = mask[i + 2] = True
            i += 3
            continue

        # ── Altitude: "<number> feet" ─────────────────────────────────────────
        if re.fullmatch(r"\d+", tok) and i + 1 < n and tokens[i + 1] == "feet":
            mask[i] = mask[i + 1] = True
            i += 2
            continue

        # ── Runway: "runway <number> [left|right|center]" ────────────────────
        if tok == "runway" and i + 1 < n and re.fullmatch(r"\d{1,2}", tokens[i + 1]):
            mask[i] = mask[i + 1] = True
            i += 2
            if i < n and tokens[i] in {"left", "right", "center"}:
                mask[i] = True
                i += 1
            continue

        # ── Frequency: "<3-digit> decimal <number>" ──────────────────────────
        if re.fullmatch(r"\d{3}", tok) and i + 2 < n \
                and tokens[i + 1] == "decimal" \
                and re.fullmatch(r"\d+", tokens[i + 2]):
            mask[i] = mask[i + 1] = mask[i + 2] = True
            i += 3
            continue

        # ── Standalone frequency: 5-digit number like "13452" ────────────────
        if re.fullmatch(r"\d{5}", tok):
            mask[i] = True
            i += 1
            continue

        # ── Heading: "heading <number>" ───────────────────────────────────────
        if tok == "heading" and i + 1 < n and re.fullmatch(r"\d+", tokens[i + 1]):
            mask[i] = mask[i + 1] = True
            i += 2
            continue

        # ── Squawk: "squawk <4-digit>" ────────────────────────────────────────
        if tok == "squawk" and i + 1 < n and re.fullmatch(r"\d{4}", tokens[i + 1]):
            mask[i] = mask[i + 1] = True
            i += 2
            continue

        # ── Phonetic callsign: 2+ consecutive ICAO phonetic words ────────────
        if tok in PHONETIC:
            j = i
            while j < n and tokens[j] in PHONETIC:
                j += 1
            # also grab trailing numbers (e.g. "oscar kilo foxtrot alfa oscar")
            while j < n and re.fullmatch(r"\d+", tokens[j]):
                j += 1
            if j - i >= 2:          # at least 2 phonetic words = callsign
                for k in range(i, j):
                    mask[k] = True
                i = j
                continue

        # ── Airline callsign: known name + flight number ──────────────────────
        if tok in AIRLINE_NAMES and i + 1 < n and re.fullmatch(r"\d+", tokens[i + 1]):
            mask[i] = mask[i + 1] = True
            i += 2
            # trailing phonetic (e.g. "ryan air 92 bravo quebec")
            while i < n and (tokens[i] in PHONETIC or re.fullmatch(r"\d+", tokens[i])):
                mask[i] = True
                i += 1
            continue

        # ── Two-word airline name + number (e.g. "air france 356") ───────────
        if tok in AIRLINE_NAMES and i + 2 < n \
                and re.fullmatch(r"[a-z]+", tokens[i + 1]) \
                and re.fullmatch(r"\d+", tokens[i + 2]):
            mask[i] = mask[i + 1] = mask[i + 2] = True
            i += 3
            continue

        i += 1

    return mask


# ── Weighted WER ──────────────────────────────────────────────────────────────

def _levenshtein_weighted(ref_tokens, hyp_tokens, ref_mask, w=ENTITY_WEIGHT):
    """
    Standard edit-distance but substitution/deletion of an entity token
    in ref costs w; insertion of any token costs 1.
    Returns (weighted_errors, weighted_total_ref_cost).
    """
    n, m = len(ref_tokens), len(hyp_tokens)

    # cost[i] = weight of ref token i
    ref_cost = [w if ref_mask[i] else 1 for i in range(n)]

    # DP table
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + ref_cost[i - 1]   # deletion
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 1                   # insertion

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                delete  = dp[i - 1][j] + ref_cost[i - 1]
                insert  = dp[i][j - 1] + 1
                replace = dp[i - 1][j - 1] + ref_cost[i - 1]
                dp[i][j] = min(delete, insert, replace)

    errors   = dp[n][m]
    total    = sum(ref_cost)
    return errors, total


def weighted_wer_corpus(references, hypotheses, w=ENTITY_WEIGHT):
    """Corpus-level weighted WER (single number, lower is better)."""
    total_errors = 0.0
    total_cost   = 0.0

    for ref, hyp in zip(references, hypotheses):
        ref_tok  = tokenise(ref)
        hyp_tok  = tokenise(hyp)
        ref_mask = entity_token_mask(ref_tok)

        if not ref_tok:
            continue

        errors, cost = _levenshtein_weighted(ref_tok, hyp_tok, ref_mask, w)
        total_errors += errors
        total_cost   += cost

    return round(total_errors / total_cost * 100, 4) if total_cost else None


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    examples = [
        # (reference, hypothesis, description)
        ("lufthansa 4393 descend to flight level 270",
         "lufthansa 4393 descend to flight level 270",
         "perfect"),
        ("lufthansa 4393 descend to flight level 270",
         "lufthansa 4393 descend to flight level 200",
         "entity error (flight level)"),
        ("delta go bird 123",
         "delta no bird 123",
         "non-entity error (go→no), entity correct"),
        ("runway 27 cleared to land",
         "runway 22 cleared to land",
         "entity error (runway)"),
        ("contact rhein 132 decimal 4",
         "contact rhein 132 decimal 9",
         "entity error (frequency)"),
        ("oscar kilo foxtrot alfa oscar taxi to holding point runway 27",
         "oscar kilo foxtrot alfa oscar taxi to holding point runway 22",
         "phonetic callsign ok, runway error"),
    ]

    print(f"{'Description':<45} {'Ref':<50} {'Hyp':<50} {'WER':>6} {'WWER':>6}")
    print("-" * 165)
    for ref, hyp, desc in examples:
        ref_tok  = tokenise(ref)
        hyp_tok  = tokenise(hyp)
        ref_mask = entity_token_mask(ref_tok)

        tagged = " ".join(f"[{t}]" if ref_mask[i] else t
                          for i, t in enumerate(ref_tok))

        plain_wer = jiwer.wer(ref, hyp) * 100
        w_wer     = weighted_wer_corpus([ref], [hyp])

        print(f"{desc:<45} WER={plain_wer:.1f}%  WWER={w_wer:.1f}%")
        print(f"  Tagged ref: {tagged}")
        print()
