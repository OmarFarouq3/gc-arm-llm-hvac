"""LLM rule classification.

Paper: Section 3.4, Fig. 10. Prompt = System / Rule / Context / User.
Output JSON: {"validity": bool, "label": 0-3, "reason": "<= 60 words"}.

Owner: Member C
"""


def build_prompt(rule, context):
    raise NotImplementedError


def classify_rule(rule, context, model, temperature, top_p):
    raise NotImplementedError


def classify_all(rules, runs):
    raise NotImplementedError
