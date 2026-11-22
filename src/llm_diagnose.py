"""LLM rule classification.

Paper: Section 3.4, Fig. 10. Prompt = System / Rule / Context / User.
Output JSON: {"validity": bool, "label": 0-3, "reason": "<= 60 words"}.

The LLM is reached through an OpenAI-compatible API (by default a local Ollama
server). Settings come from .env (LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY) and
config.yaml (llm.temperature, llm.top_p, llm.runs, llm.max_reason_words).
LLM_MODEL in .env overrides llm.model in config.yaml.

Reasoning ("thinking") models such as Qwen 3.5: we send reasoning_effort="none".
Measured on Ollama 0.40.16 with qwen3.5:4b, this is the only option that turns
thinking off on the OpenAI-compatible endpoint (about 1 s instead of about 20 s per call);
extra_body={"think": False} and "/no_think" in the message were both ignored.
Any <think>...</think> block that still appears in the answer is stripped before parsing.

Owner: Member C
"""

import json
import os
import re
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tables"

load_dotenv(ROOT / ".env")
_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

SETTINGS = {
    "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    "api_key": os.getenv("OPENAI_API_KEY") or "ollama",   # Ollama ignores the key
    "model": os.getenv("LLM_MODEL") or _CONFIG["llm"]["model"],
    "temperature": _CONFIG["llm"]["temperature"],
    "top_p": _CONFIG["llm"]["top_p"],
    "runs": _CONFIG["llm"]["runs"],
    "max_reason_words": _CONFIG["llm"]["max_reason_words"],
}

# Extra request arguments that switch reasoning off (see module docstring).
NO_THINKING = {"reasoning_effort": "none"}

LABELS = {
    0: "0 Invalid rules",
    1: "1 Design anomaly",
    2: "2 Operational anomaly",
    3: "3 Performance degradation",
}

SYSTEM_PROMPT = ("You are an expert in the HVAC field and now need to evaluate the "
                 "validity and category of the mined association rules.")

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=SETTINGS["base_url"], api_key=SETTINGS["api_key"])
    return _client


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def rule_to_text(rule):
    """Rule dict (format from src/gcarm.py) -> one readable line, e.g.
    "Antecedent: CT_SW_TEMP_1=high; Consequent: CT_FAN_SPD_1=high; Support: 0.0500;
     Confidence: 0.6200; Lift: 1.40". Several items are joined with " AND ".
    Metrics that are missing or None are left out."""
    parts = [f"Antecedent: {' AND '.join(rule['antecedent'])}",
             f"Consequent: {' AND '.join(rule['consequent'])}"]
    for key, fmt in (("support", "{:.4f}"), ("confidence", "{:.4f}"), ("lift", "{:.2f}")):
        if rule.get(key) is not None:
            parts.append(f"{key.capitalize()}: {fmt.format(rule[key])}")
    return "; ".join(parts)


def _context_lines(context):
    """Accept a string, a list of strings, or a list of retrieved chunk dicts."""
    if not context:
        return []
    if isinstance(context, str):
        return [context]
    return [c["text"] if isinstance(c, dict) else str(c) for c in context]


def build_prompt(rule_text, context, extra_lines=None):
    """The paper's 4-part prompt as chat messages.

    [System] -> system message. [Rule], [Context], [User] -> one user message.
    context:     string, list of strings, or list of chunks from rag.retrieve();
                 empty/None gives "[Context] None" (the no-knowledge setting).
    extra_lines: optional list of extra strings appended to [Context] as bullets.
                 Hook for Member D (e.g. confounder evidence); no other effect.
    Returns a list of {"role", "content"} dicts ready for the chat API.
    """
    lines = _context_lines(context) + list(extra_lines or [])
    context_block = "\n".join(f"- {line}" for line in lines) if lines else "None"
    labels = ", ".join(f'"{v}"' for v in LABELS.values())
    user = (
        f"[Rule]\n{rule_text}\n\n"
        f"[Context]\n{context_block}\n\n"
        "[User]\nPlease reply in JSON with three fields: "
        '1. "validity": true or false; '
        f'2. "label": one of {{{labels}}}; '
        f'3. "reason": concise (<={SETTINGS["max_reason_words"]} words) explanation. '
        # Added after the first test: the model read "validity" as "the rule is true".
        "validity is true only if the rule indicates an anomaly (labels 1-3); "
        "for label 0 it is false."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Parsing the answer
# ---------------------------------------------------------------------------

def _strip_think(text):
    """Remove <think>...</think> blocks (and an unclosed leading <think>)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "<think>" in text:                    # thinking cut off before </think>
        text = text.split("<think>")[0]
    return text.strip()


def _extract_json(text):
    """Parse JSON from the answer; fall back to the first {...} block in plain text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))      # may raise; caught by the caller
        raise


def _parse_validity(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"bad validity: {value!r}")


def _parse_label(value):
    """Accept 2, "2", "2 Operational anomaly" or "Operational anomaly" -> 2."""
    if isinstance(value, int) and value in LABELS:
        return value
    s = str(value).strip()
    m = re.match(r"^([0-3])\b", s)
    if m:
        return int(m.group(1))
    for k, name in LABELS.items():
        if name[2:].lower().rstrip("s") in s.lower():
            return k
    raise ValueError(f"bad label: {value!r}")


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------

def classify_rule(prompt, temperature=None, top_p=None, model=None):
    """Send one prompt (from build_prompt) to the LLM and parse the answer.

    Never raises. Returns a dict with:
      validity (bool or None), label (int 0-3 or None), reason (str),
      failed (bool), error (str), raw (raw answer text), latency_s,
      prompt_tokens, completion_tokens, total_tokens.
    """
    out = {"validity": None, "label": None, "reason": "", "failed": True, "error": "",
           "raw": "", "latency_s": None, "prompt_tokens": None,
           "completion_tokens": None, "total_tokens": None}
    t0 = time.perf_counter()
    try:
        resp = _get_client().chat.completions.create(
            model=model or SETTINGS["model"],
            messages=prompt,
            temperature=SETTINGS["temperature"] if temperature is None else temperature,
            top_p=SETTINGS["top_p"] if top_p is None else top_p,
            response_format={"type": "json_object"},     # ask for JSON mode
            **NO_THINKING,
        )
    except Exception as e:                               # server down, bad model name, ...
        out.update(latency_s=time.perf_counter() - t0, error=f"API error: {e}")
        return out

    out["latency_s"] = time.perf_counter() - t0
    if resp.usage is not None:
        out.update(prompt_tokens=resp.usage.prompt_tokens,
                   completion_tokens=resp.usage.completion_tokens,
                   total_tokens=resp.usage.total_tokens)
    out["raw"] = resp.choices[0].message.content or ""

    try:
        data = _extract_json(_strip_think(out["raw"]))
        out.update(validity=_parse_validity(data["validity"]),
                   label=_parse_label(data["label"]),
                   reason=str(data.get("reason", "")).strip(),
                   failed=False)
    except Exception as e:                               # bad JSON, missing field, ...
        out["error"] = f"parse error: {type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# Many rules
# ---------------------------------------------------------------------------

def classify_all(rules, runs=None, use_rag=True, extra_lines_fn=None, out_name=None,
                 temperature=None, top_p=None, verbose=True):
    """Classify every rule `runs` times; one DataFrame row per (rule, run).

    use_rag=True:   context = variable definitions from the glossary + top-k chunks
                    retrieved from the knowledge base (src/rag.py).
    use_rag=False:  empty context (baseline without knowledge).
    extra_lines_fn: optional function rule -> list of strings added to [Context]
                    (Member D's hook, e.g. confounder evidence).
    The table is saved to results/tables/<out_name>.csv
    (default "llm_rag" or "llm_norag"). Returns the DataFrame.
    """
    import pandas as pd
    from tqdm.auto import tqdm

    runs = runs or SETTINGS["runs"]
    if use_rag:
        from src import rag

    rows = []
    for rule_id, rule in enumerate(tqdm(rules, disable=not verbose, desc="rules")):
        rule_text = rule_to_text(rule)
        # Retrieval is deterministic, so it is done once per rule, not per run.
        if use_rag:
            chunks = rag.retrieve(rule_text)
            context = rag.describe_variables(rule_text) + [c["text"] for c in chunks]
            sources = "; ".join(f"{c['source']}#{c['id']}" for c in chunks)
        else:
            context, sources = [], ""
        extra = extra_lines_fn(rule) if extra_lines_fn else None
        prompt = build_prompt(rule_text, context, extra)

        for run in range(1, runs + 1):
            result = classify_rule(prompt, temperature, top_p)
            rows.append({
                "rule_id": rule_id, "run": run,
                "antecedent": " AND ".join(rule["antecedent"]),
                "consequent": " AND ".join(rule["consequent"]),
                "support": rule.get("support"), "confidence": rule.get("confidence"),
                "lift": rule.get("lift"),
                "use_rag": use_rag, "context_sources": sources,
                "model": SETTINGS["model"],
                "temperature": SETTINGS["temperature"] if temperature is None else temperature,
                "top_p": SETTINGS["top_p"] if top_p is None else top_p,
                **result,
            })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{out_name or ('llm_rag' if use_rag else 'llm_norag')}.csv"
    df.to_csv(path, index=False)
    if verbose:
        print(f"saved {len(df)} rows to {path.relative_to(ROOT)}  "
              f"(failed: {int(df['failed'].sum())})")
    return df
